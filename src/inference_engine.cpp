#include "ryzenai/inference_engine.h"
#include <ort_genai.h>
#include <ort_genai_c.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <iomanip>
#include <ctime>

namespace ryzenai {

namespace fs = std::filesystem;

namespace {

void writeAppendDebugFile(const std::string& label,
                          const std::string& session_id,
                          size_t turn_number,
                          const std::string& text) {
    const char* debug_dir_env = std::getenv("AMDGPU_DEBUG_DIR");
    if (!debug_dir_env || debug_dir_env[0] == '\0') {
        return;
    }

    try {
        fs::path dir(debug_dir_env);
        fs::create_directories(dir);

        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::time_t tt = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm{};
        localtime_s(&local_tm, &tt);

        std::ostringstream name;
        name << "server_append_" << std::put_time(&local_tm, "%Y%m%d_%H%M%S")
             << '_' << std::setw(3) << std::setfill('0') << ms.count()
             << ".txt";

        fs::path out = dir / name.str();
        std::ofstream file(out);
        file << "label=" << label << "\n";
        file << "session=" << session_id << "\n";
        file << "turn=" << turn_number << "\n";
        file << "chars=" << text.length() << "\n";
        file << "----text----\n";
        file << text;
        std::cout << "[InferenceEngine] Wrote append debug: " << out.string() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to write append debug file: " << e.what() << std::endl;
    }
}

size_t countSubstring(const std::string& haystack, const std::string& needle) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

void logMultiTurnAppendPlan(const char* path_label,
                            const std::string& conversation_id,
                            size_t turn_number,
                            bool is_first_turn,
                            const std::string& full_prompt,
                            const std::string& text_to_append,
                            int full_tokens,
                            int append_tokens,
                            const char* input_mode) {
    static constexpr const char* kUserTurnMarker = "<|im_start|>user";
    const size_t user_markers = countSubstring(full_prompt, kUserTurnMarker);
    const size_t full_chars = full_prompt.size();
    const size_t append_chars = text_to_append.size();
    const int saved_tokens = full_tokens - append_tokens;
    const int saved_pct = (full_tokens > 0) ? static_cast<int>((saved_tokens * 100LL) / full_tokens) : 0;

    std::cout << "[InferenceEngine][MultiTurnDebug] path=" << path_label
              << " session=" << conversation_id
              << " turn=" << turn_number
              << " mode=" << (is_first_turn ? "APPEND_FULL" : "APPEND_DELTA")
              << " input=" << input_mode
              << " user_markers=" << user_markers
              << " full_chars=" << full_chars
              << " full_tokens=" << full_tokens
              << " append_chars=" << append_chars
              << " append_tokens=" << append_tokens;
    if (!is_first_turn) {
        std::cout << " saved_tokens=" << saved_tokens << " saved_pct=" << saved_pct << "%";
    }
    std::cout << std::endl;

    if (!is_first_turn) {
        std::cout << "[InferenceEngine][MultiTurnDebug] delta_head: "
                  << text_to_append.substr(0, std::min(size_t(240), append_chars)) << std::endl;
    }
}

void logMultiTurnAfterAppend(const char* path_label,
                             size_t seq_before,
                             size_t seq_after,
                             bool is_done) {
    std::cout << "[InferenceEngine][MultiTurnDebug] path=" << path_label
              << " seq_before=" << seq_before
              << " seq_after=" << seq_after
              << " seq_delta=" << (seq_after > seq_before ? seq_after - seq_before : 0)
              << " is_done=" << (is_done ? "true" : "false") << std::endl;
    if (is_done && seq_after <= seq_before) {
        std::cerr << "[InferenceEngine][MultiTurnDebug] WARNING: generator IsDone after append "
                  << "with no new sequence tokens — generation loop will produce 0 output"
                  << std::endl;
    }
}

void logMultiTurnCompleted(const char* path_label,
                           size_t turn_number,
                           size_t seq_before,
                           size_t seq_after,
                           int generated_tokens,
                           int streamed_tokens) {
    std::cout << "[InferenceEngine][MultiTurnDebug] path=" << path_label
              << " turn=" << turn_number
              << " completed generated_tokens=" << generated_tokens
              << " seq_before=" << seq_before
              << " seq_after=" << seq_after;
    if (streamed_tokens >= 0) {
        std::cout << " streamed_tokens=" << streamed_tokens;
    }
    if (generated_tokens == 0) {
        std::cout << " WARNING=zero_output";
    }
    std::cout << std::endl;
}

}  // namespace

InferenceEngine::InferenceEngine(const std::string& model_path, int ctx_size)
    : ctx_size_(ctx_size) {

    std::cout << "[InferenceEngine] Initializing with model: " << model_path << std::endl;

    // Resolve model path (handles Hugging Face cache structure)
    model_path_ = resolveModelPath(model_path);
    if (model_path_ != model_path) {
        std::cout << "[InferenceEngine] Resolved to: " << model_path_ << std::endl;
    }
    
    // Validate model directory
    if (!validateModelDirectory(model_path_)) {
        throw std::runtime_error("Invalid model directory: " + model_path_);
    }
    
    // Auto-detect execution mode from genai_config.json
    execution_mode_ = detectExecutionMode();
    std::cout << "[InferenceEngine] Detected execution mode: " << execution_mode_ << std::endl;
    
    // Detect Ryzen AI version and load config
    loadRaiConfig();

    // Establish the context window. Read the model's native context_length, then
    // pick: explicit --ctx-size (clamped to native) > native context_length.
    // This replaces the legacy 2048 default unless rai_config.json pinned a value.
    try {
        std::string cfg = model_path_ + "/genai_config.json";
        if (fs::exists(cfg)) {
            std::ifstream f(cfg);
            json c = json::parse(f);
            model_context_length_ = c["model"].value("context_length", 0);
        }
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Could not read context_length: " << e.what() << std::endl;
    }
    if (!rai_max_prompt_set_) {
        if (ctx_size_ > 0) {
            max_prompt_length_ = (model_context_length_ > 0)
                ? std::min(ctx_size_, model_context_length_) : ctx_size_;
        } else if (model_context_length_ > 0) {
            max_prompt_length_ = model_context_length_;
        }
    }
    std::cout << "[InferenceEngine] Context window: model_context_length="
              << model_context_length_ << ", requested ctx_size=" << ctx_size_
              << ", effective max_prompt_length=" << max_prompt_length_ << std::endl;

    // Setup execution provider
    setupExecutionProvider();
    
    // Load the model
    loadModel();
    
    // Extract model name from path
    model_name_ = fs::path(model_path_).filename().string();
    
    // Load default generation params from genai_config.json
    std::string config_path = model_path_ + "/genai_config.json";
    if (fs::exists(config_path)) {
        try {
            std::ifstream file(config_path);
            json config = json::parse(file);
            
            if (config.contains("search")) {
                json search = config["search"];
                has_search_config_ = true;
                
                // Load defaults from search config (matching Python implementation)
                // NOTE: search.max_length from genai_config.json means TOTAL sequence length,
                // but default_params_.max_length is used as "max NEW tokens" in our code.
                // We intentionally DON'T load search.max_length here to avoid semantic confusion.
                // The user's max_tokens parameter will be used directly as max new tokens.
                
                if (search.contains("temperature") && search["temperature"].is_number()) {
                    default_params_.temperature = search["temperature"];
                }
                if (search.contains("top_p") && search["top_p"].is_number()) {
                    default_params_.top_p = search["top_p"];
                }
                if (search.contains("top_k") && search["top_k"].is_number()) {
                    default_params_.top_k = search["top_k"];
                }
                if (search.contains("repetition_penalty") && search["repetition_penalty"].is_number()) {
                    default_params_.repetition_penalty = search["repetition_penalty"];
                }
                if (search.contains("do_sample") && search["do_sample"].is_boolean()) {
                    default_params_.do_sample = search["do_sample"];
                }
                
                std::cout << "[InferenceEngine] Loaded search config from genai_config.json" << std::endl;
                std::cout << "  - temperature: " << default_params_.temperature << std::endl;
                std::cout << "  - top_p: " << default_params_.top_p << std::endl;
                std::cout << "  - top_k: " << default_params_.top_k << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Failed to load search config from genai_config.json: " << e.what() << std::endl;
        }
    }
    
    std::cout << "[InferenceEngine] Model loaded successfully: " << model_name_ << std::endl;
    std::cout << "[InferenceEngine] Max prompt length: " << max_prompt_length_ << " tokens" << std::endl;
}

InferenceEngine::~InferenceEngine() {
    std::cout << "[InferenceEngine] Shutting down" << std::endl;
}

GenerationParams InferenceEngine::getDefaultParams() const {
    return default_params_;
}

std::string InferenceEngine::applyChatTemplate(const std::string& messages_json,
                                               const std::string& tools_json,
                                               bool add_generation_prompt) {
    // Parse messages
    json messages = json::parse(messages_json);
    std::ostringstream prompt;
    
    // Check if we have a Qwen-style chat template (contains <|im_start|>)
    bool is_qwen_style = !chat_template_.empty() && 
                         (chat_template_.find("<|im_start|>") != std::string::npos ||
                          chat_template_.find("\\u003c|im_start|\\u003e") != std::string::npos);
    
    // If tools are provided, always use OGA's built-in template (it handles tools properly)
    if (!tools_json.empty()) {
        try {
            const char* template_str = chat_template_.empty() ? nullptr : chat_template_.c_str();
            const char* tools_str = tools_json.c_str();
            
            auto result = tokenizer_->ApplyChatTemplate(
                template_str,
                messages_json.c_str(),
                tools_str,
                add_generation_prompt
            );
            
            std::cout << "[InferenceEngine] Applied chat template with tools" << std::endl;
            return std::string(result);
            
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to apply chat template with tools: " << e.what() << std::endl;
            throw;
        }
    } else if (is_qwen_style) {
        // Use Qwen/ChatML format: <|im_start|>role\ncontent<|im_end|>\n
        for (const auto& msg : messages) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            
            prompt << "<|im_start|>" << role << "\n"
                   << content << "<|im_end|>\n";
        }
        
        if (add_generation_prompt) {
            prompt << "<|im_start|>assistant\n";
        }
        
        std::cout << "[InferenceEngine] Applied Qwen/ChatML template" << std::endl;
    } else {
        // Try using the OGA's built-in chat template
        try {
            const char* template_str = chat_template_.empty() ? nullptr : chat_template_.c_str();
            
            auto result = tokenizer_->ApplyChatTemplate(
                template_str,
                messages_json.c_str(),
                nullptr,
                add_generation_prompt
            );
            
            return std::string(result);
            
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] OGA chat template failed: " << e.what() << std::endl;
            std::cerr << "[WARNING] Using simple fallback template" << std::endl;
            
            // Simple fallback template
            prompt.str("");  // Clear
            for (const auto& msg : messages) {
                std::string role = msg.value("role", "user");
                std::string content = msg.value("content", "");
                
                if (role == "system") {
                    prompt << "System: " << content << "\n\n";
                } else if (role == "user") {
                    prompt << "User: " << content << "\n\n";
                } else if (role == "assistant") {
                    prompt << "Assistant: " << content << "\n\n";
                }
            }
            
            if (add_generation_prompt) {
                prompt << "Assistant: ";
            }
        }
    }
    
    return prompt.str();
}

std::string InferenceEngine::resolveModelPath(const std::string& path) {
    // If path has a "snapshots" subdirectory (Hugging Face cache structure),
    // automatically find the latest snapshot
    std::string snapshots_dir = path + "/snapshots";
    if (fs::exists(snapshots_dir) && fs::is_directory(snapshots_dir)) {
        std::cout << "[InferenceEngine] Detected Hugging Face cache structure, looking for snapshot..." << std::endl;
        
        // Find the first (and usually only) snapshot directory
        for (const auto& entry : fs::directory_iterator(snapshots_dir)) {
            if (entry.is_directory()) {
                std::string snapshot_path = entry.path().string();
                std::cout << "[InferenceEngine] Found snapshot: " << snapshot_path << std::endl;
                return snapshot_path;
            }
        }
        
        std::cerr << "[ERROR] No snapshot found in: " << snapshots_dir << std::endl;
        return path;
    }
    
    // Otherwise, use the path as-is
    return path;
}

bool InferenceEngine::validateModelDirectory(const std::string& path) {
    if (!fs::exists(path) || !fs::is_directory(path)) {
        std::cerr << "[ERROR] Model path does not exist or is not a directory: " << path << std::endl;
        return false;
    }
    
    // Check for required files (at minimum genai_config.json)
    std::string config_path = path + "/genai_config.json";
    if (!fs::exists(config_path)) {
        std::cerr << "[ERROR] Required file not found: " << config_path << std::endl;
        return false;
    }
    
    return true;
}

std::string InferenceEngine::detectRyzenAIVersion() {
    // Priority 1: Check RYZENAI_VERSION environment variable directly
    const char* version_env = std::getenv("RYZENAI_VERSION");
    if (version_env) {
        return std::string(version_env);
    }

    // Priority 2: Check RYZENAI_INSTALL_PATH environment variable and extract version
    const char* install_path_env = std::getenv("RYZENAI_INSTALL_PATH");
    if (install_path_env) {
        std::string path_str(install_path_env);
        // Remove trailing slashes
        while (!path_str.empty() && (path_str.back() == '/' || path_str.back() == '\\')) {
            path_str.pop_back();
        }
        // Extract version from path (e.g., "/opt/ryzenai/1.7.0" -> "1.7.0")
        size_t last_slash = path_str.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            std::string version = path_str.substr(last_slash + 1);
            // Validate it looks like a version number: starts with digit and contains a dot
            if (!version.empty() && std::isdigit(version[0]) && version.find('.') != std::string::npos) {
                return version;
            }
        }
    }

    // Priority 3: Check platform-specific default paths
#ifdef _WIN32
    std::string ryzenai_path_17 = "C:/Program Files/RyzenAI/1.7.0";
#else
    std::string ryzenai_path_17 = "/opt/ryzenai/1.7.0";
#endif

    if (fs::exists(ryzenai_path_17)) {
        return "1.7.0";
    }

    // Default fallback
    return "1.7.0";
}

std::string InferenceEngine::detectExecutionMode() {
    // Auto-detect execution mode from genai_config.json by inspecting
    // session_options in model.decoder. The key indicators are:
    //
    // NPU:    "hybrid_opt_token_backend": "npu" in either config_entries
    //         or provider_options[].RyzenAI
    // Hybrid: provider_options[].RyzenAI exists (without token_backend=npu)
    // CPU:    empty provider_options, no config_entries
    
    std::string config_path = model_path_ + "/genai_config.json";
    if (!fs::exists(config_path)) {
        std::cout << "[InferenceEngine] No genai_config.json found, defaulting to cpu mode" << std::endl;
        return "cpu";
    }
    
    try {
        std::ifstream file(config_path);
        json config = json::parse(file);
        
        auto& session_opts = config["model"]["decoder"]["session_options"];

        // Check provider_options for the AMDGPU provider (hipep EP, gfx GPU).
        // genai_config.json selects this EP via provider_options:[{"AMDGPU":{...}}].
        if (session_opts.contains("provider_options") &&
            session_opts["provider_options"].is_array()) {
            for (const auto& provider : session_opts["provider_options"]) {
                if (provider.contains("AMDGPU")) {
                    return "gpu";
                }
            }
        }

        // Check config_entries for hybrid_opt_token_backend == "npu"
        if (session_opts.contains("config_entries")) {
            auto& entries = session_opts["config_entries"];
            if (entries.contains("hybrid_opt_token_backend") &&
                entries["hybrid_opt_token_backend"] == "npu") {
                return "npu";
            }
        }
        
        // Check provider_options for RyzenAI configuration
        if (session_opts.contains("provider_options") && 
            session_opts["provider_options"].is_array()) {
            for (const auto& provider : session_opts["provider_options"]) {
                if (provider.contains("RyzenAI")) {
                    auto& ryzenai = provider["RyzenAI"];
                    // If RyzenAI has hybrid_opt_token_backend == "npu", it's NPU
                    if (ryzenai.contains("hybrid_opt_token_backend") &&
                        ryzenai["hybrid_opt_token_backend"] == "npu") {
                        return "npu";
                    }
                    // Otherwise RyzenAI provider present means hybrid
                    return "hybrid";
                }
            }
        }
        
        // No RyzenAI provider, no NPU config_entries → CPU
        return "cpu";
        
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to detect execution mode from genai_config.json: " 
                  << e.what() << std::endl;
        std::cerr << "[WARNING] Defaulting to cpu mode" << std::endl;
        return "cpu";
    }
}

void InferenceEngine::loadRaiConfig() {
    // Detect Ryzen AI version
    ryzenai_version_ = detectRyzenAIVersion();
    std::cout << "[InferenceEngine] Ryzen AI version: " << ryzenai_version_ << std::endl;
    
    // Load rai_config.json if it exists
    std::string rai_config_path = model_path_ + "/rai_config.json";
    if (fs::exists(rai_config_path)) {
        try {
            std::ifstream file(rai_config_path);
            json config = json::parse(file);
            
            if (config.contains("max_prompt_length") && 
                config["max_prompt_length"].contains(ryzenai_version_)) {
                max_prompt_length_ = config["max_prompt_length"][ryzenai_version_];
                rai_max_prompt_set_ = true;
                std::cout << "[InferenceEngine] Loaded max_prompt_length from rai_config.json: "
                         << max_prompt_length_ << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Failed to parse rai_config.json: " << e.what() << std::endl;
        }
    }
}

void InferenceEngine::setupExecutionProvider() {
    std::cout << "[InferenceEngine] Setting up execution provider for mode: " << execution_mode_ << std::endl;
    
    // Note: Actual execution provider configuration happens in ONNX Runtime GenAI
    // based on the genai_config.json file. This method is mainly for validation.
    
    if (execution_mode_ == "npu") {
        std::cout << "[InferenceEngine] Using NPU (VitisAI) execution provider" << std::endl;
    } else if (execution_mode_ == "hybrid") {
        std::cout << "[InferenceEngine] Using Hybrid (NPU + iGPU) execution provider" << std::endl;
    } else if (execution_mode_ == "cpu") {
        std::cout << "[InferenceEngine] Using CPU execution provider" << std::endl;
    }
}

void InferenceEngine::loadModel() {
    try {
        std::cout << "[InferenceEngine] Loading ONNX model from: " << model_path_ << std::endl;
        
        // Create model using factory method
        model_ = OgaModel::Create(model_path_.c_str());
        
        // Create tokenizer using factory method
        tokenizer_ = OgaTokenizer::Create(*model_);
        
        // Load chat template from tokenizer_config.json
        std::string tokenizer_config_path = model_path_ + "/tokenizer_config.json";
        if (fs::exists(tokenizer_config_path)) {
            try {
                std::ifstream file(tokenizer_config_path);
                json config = json::parse(file);
                if (config.contains("chat_template") && config["chat_template"].is_string()) {
                    chat_template_ = config["chat_template"];
                    std::cout << "[InferenceEngine] Loaded chat template from tokenizer_config.json" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[WARNING] Failed to load chat template: " << e.what() << std::endl;
            }
        }

        // Many exported VLMs (e.g. Qwen-VL) ship the template as a standalone
        // chat_template.jinja rather than inline in tokenizer_config.json.
        if (chat_template_.empty()) {
            std::string jinja_path = model_path_ + "/chat_template.jinja";
            if (fs::exists(jinja_path)) {
                std::ifstream jf(jinja_path);
                std::stringstream ss;
                ss << jf.rdbuf();
                chat_template_ = ss.str();
                std::cout << "[InferenceEngine] Loaded chat template from chat_template.jinja" << std::endl;
            }
        }

        // Detect and initialize multimodal (vision) pipeline if present.
        detectMultimodal();

        std::cout << "[InferenceEngine] Model and tokenizer loaded successfully" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

std::vector<int32_t> InferenceEngine::truncatePrompt(const std::vector<int32_t>& input_ids) {
    if (input_ids.size() <= static_cast<size_t>(max_prompt_length_)) {
        return input_ids;
    }
    
    // Truncate from the beginning to keep the most recent context
    size_t truncate_amount = input_ids.size() - max_prompt_length_;
    std::cout << "[WARNING] Prompt exceeds maximum length (" 
              << input_ids.size() << " > " << max_prompt_length_ 
              << "). Truncating " << truncate_amount << " tokens from the beginning."
              << std::endl;
    
    return std::vector<int32_t>(
        input_ids.begin() + truncate_amount, 
        input_ids.end()
    );
}

std::string InferenceEngine::complete(const std::string& prompt, const GenerationParams& params, CompletionTimingData* out_timing) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    try {
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();
        auto first_token_time = start_time;
        bool first_token_received = false;
        
        // Tokenize input
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(prompt.c_str(), *sequences);
        
        // Get token IDs and apply truncation
        const int32_t* input_ids_ptr = sequences->SequenceData(0);
        size_t input_ids_count = sequences->SequenceCount(0);
        std::vector<int32_t> input_ids(input_ids_ptr, input_ids_ptr + input_ids_count);
        input_ids = truncatePrompt(input_ids);
        
        // Create generator params
        auto gen_params = OgaGeneratorParams::Create(*model_);
        // max_length should be prompt_length + max_new_tokens
        // params.max_length is max_new_tokens from the caller
        int total_max_length = static_cast<int>(input_ids.size()) + params.max_length;
        if (model_context_length_ > 0 && total_max_length > model_context_length_) {
            total_max_length = model_context_length_;
        }
        gen_params->SetSearchOption("max_length", total_max_length);
        gen_params->SetSearchOption("temperature", params.temperature);
        gen_params->SetSearchOption("top_p", params.top_p);
        gen_params->SetSearchOption("top_k", static_cast<double>(params.top_k));
        gen_params->SetSearchOption("repetition_penalty", params.repetition_penalty);
        gen_params->SetSearchOptionBool("do_sample", params.do_sample);
        // Lock random_seed to 1 for deterministic behavior (matching Python reference)
        gen_params->SetSearchOption("random_seed", 1.0);
        
        // Generate
        auto generator = OgaGenerator::Create(*model_, *gen_params);
        
        // Set input tokens
        generator->AppendTokens(input_ids.data(), input_ids.size());
        
        std::cout << "[InferenceEngine] Generating tokens..." << std::endl;
        
        while (!generator->IsDone()) {
            generator->GenerateNextToken();
            
            // Track time to first token
            if (!first_token_received) {
                first_token_time = std::chrono::high_resolution_clock::now();
                first_token_received = true;
            }
        }
        
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Get the output
        const int32_t* output_ptr = generator->GetSequenceData(0);
        size_t output_count = generator->GetSequenceCount(0);
        
        // Calculate actual generated token count
        int generated_token_count = (output_count > input_ids.size()) 
            ? static_cast<int>(output_count - input_ids.size()) 
            : 0;
        
        // Calculate timing metrics
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto ttft_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        double ttft_seconds = ttft_duration.count() / 1000.0;
        double total_time_ms = static_cast<double>(total_duration.count());
        
        // Calculate TPS: tokens generated after the first token, divided by time after first token
        double decode_time_seconds = (total_duration.count() - ttft_duration.count()) / 1000.0;
        double tps = 0.0;
        if (generated_token_count > 1 && decode_time_seconds > 0) {
            // TPS = (tokens - 1) / decode_time (exclude first token from TPS calculation)
            tps = (generated_token_count - 1) / decode_time_seconds;
        } else if (generated_token_count == 1 && total_time_ms > 0) {
            // Only one token generated - use total time
            tps = 1.0 / (total_time_ms / 1000.0);
        }
        
        // Return timing data if requested
        if (out_timing != nullptr) {
            out_timing->token_count = generated_token_count;
            out_timing->ttft_seconds = ttft_seconds;
            out_timing->tps = tps;
            out_timing->total_time_ms = total_time_ms;
        }
        
        // Decode only the newly generated tokens (skip the input prompt)
        std::string result;
        if (output_count > input_ids.size()) {
            auto decoded = tokenizer_->Decode(output_ptr + input_ids.size(), output_count - input_ids.size());
            result = std::string(decoded);
        } else {
            // No new tokens generated
            result = "";
        }
        
        // Apply stop sequences - remove stop sequence and everything after it
        for (const auto& stop_seq : params.stop_sequences) {
            size_t pos = result.find(stop_seq);
            if (pos != std::string::npos) {
                result = result.substr(0, pos);
                break;  // Stop at first match
            }
        }
        
        std::cout << "[InferenceEngine] Generated " << generated_token_count << " tokens in " 
                  << total_time_ms << "ms (TTFT: " << (ttft_seconds * 1000) << "ms, TPS: " << tps << ")" << std::endl;
        
        return result;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Inference failed: " + std::string(e.what()));
    }
}

void InferenceEngine::streamComplete(const std::string& prompt, 
                                     const GenerationParams& params,
                                     StreamCallback callback) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    
    try {
        // Tokenize input
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(prompt.c_str(), *sequences);
        
        // Get token IDs and apply truncation
        const int32_t* input_ids_ptr = sequences->SequenceData(0);
        size_t input_ids_count = sequences->SequenceCount(0);
        std::vector<int32_t> input_ids(input_ids_ptr, input_ids_ptr + input_ids_count);
        input_ids = truncatePrompt(input_ids);
        
        // Create generator params
        auto gen_params = OgaGeneratorParams::Create(*model_);
        // max_length should be prompt_length + max_new_tokens
        // params.max_length is max_new_tokens from the caller
        int total_max_length = static_cast<int>(input_ids.size()) + params.max_length;
        if (model_context_length_ > 0 && total_max_length > model_context_length_) {
            total_max_length = model_context_length_;
        }
        std::cout << "[InferenceEngine::streamComplete] prompt_length=" << input_ids.size()
                  << ", max_new_tokens=" << params.max_length
                  << ", total_max_length=" << total_max_length << std::endl;
        gen_params->SetSearchOption("max_length", total_max_length);
        gen_params->SetSearchOption("temperature", params.temperature);
        gen_params->SetSearchOption("top_p", params.top_p);
        gen_params->SetSearchOption("top_k", static_cast<double>(params.top_k));
        gen_params->SetSearchOption("repetition_penalty", params.repetition_penalty);
        gen_params->SetSearchOptionBool("do_sample", params.do_sample);
        // Lock random_seed to 1 for deterministic behavior (matching Python reference)
        gen_params->SetSearchOption("random_seed", 1.0);
        
        // Generate
        auto generator = OgaGenerator::Create(*model_, *gen_params);
        
        // Set input tokens
        generator->AppendTokens(input_ids.data(), input_ids.size());
        
        std::cout << "[InferenceEngine] Generating tokens (streaming)..." << std::endl;
        
        // Use OgaTokenizerStream for efficient incremental token decoding
        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer_);
        
        size_t token_count = 0;
        std::string accumulated_output;  // Track full output for stop sequence detection
        bool client_disconnected = false;  // Track if client disconnected
        
        while (!generator->IsDone() && !client_disconnected) {
            generator->GenerateNextToken();
            
            // Get just the new token
            const int32_t* all_tokens = generator->GetSequenceData(0);
            size_t num_tokens = generator->GetSequenceCount(0);
            int32_t new_token = all_tokens[num_tokens - 1];
            
            // Decode incrementally using tokenizer stream (this works!)
            const char* decoded = tokenizer_stream->Decode(new_token);
            if (decoded && decoded[0] != '\0') {
                std::string token_str(decoded);
                
                // Check for stop sequences before accumulating
                bool should_stop = false;
                for (const auto& stop_seq : params.stop_sequences) {
                    // Check if adding this token would complete a stop sequence
                    std::string temp_output = accumulated_output + token_str;
                    if (temp_output.find(stop_seq) != std::string::npos) {
                        should_stop = true;
                        break;
                    }
                }
                
                if (should_stop) {
                    // Stop generation - don't send this token
                    break;
                }
                
                accumulated_output += token_str;
                bool is_final = generator->IsDone();
                
                // Call callback and check if client is still connected
                if (!callback(token_str, is_final)) {
                    client_disconnected = true;
                    std::cout << "[InferenceEngine] Client disconnected, stopping generation" << std::endl;
                    break;
                }
            }
            
            token_count++;
        }
        
        if (client_disconnected) {
            std::cout << "[InferenceEngine] Generation stopped early due to client disconnect (generated " 
                      << token_count << " tokens)" << std::endl;
        }
        
        std::cout << "[InferenceEngine] Generated " << token_count << " tokens (streaming)" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Streaming inference failed: " + std::string(e.what()));
    }
}

int InferenceEngine::countTokens(const std::string& text) {
    try {
        auto sequences = OgaSequences::Create();
        tokenizer_->Encode(text.c_str(), *sequences);
        return static_cast<int>(sequences->SequenceCount(0));
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to count tokens: " << e.what() << std::endl;
        return 0;
    }
}

std::vector<int32_t> InferenceEngine::encodeText(const std::string& text) const {
    auto sequences = OgaSequences::Create();
    tokenizer_->Encode(text.c_str(), *sequences);
    const int32_t* input_ids_ptr = sequences->SequenceData(0);
    size_t input_ids_count = sequences->SequenceCount(0);
    return std::vector<int32_t>(input_ids_ptr, input_ids_ptr + input_ids_count);
}

void InferenceEngine::appendPromptText(OgaGenerator& generator, const std::string& text) {
    if (text.empty()) {
        return;
    }
    std::vector<int32_t> input_ids = encodeText(text);
    generator.AppendTokens(input_ids.data(), input_ids.size());
}

void InferenceEngine::configureGeneratorParams(OgaGeneratorParams& gen_params,
                                               const GenerationParams& params,
                                               int total_max_length) const {
    gen_params.SetSearchOption("max_length", total_max_length);
    gen_params.SetSearchOption("temperature", params.temperature);
    gen_params.SetSearchOption("top_p", params.top_p);
    gen_params.SetSearchOption("top_k", static_cast<double>(params.top_k));
    gen_params.SetSearchOption("repetition_penalty", params.repetition_penalty);
    gen_params.SetSearchOptionBool("do_sample", params.do_sample);
    gen_params.SetSearchOption("random_seed", 1.0);
}

std::string InferenceEngine::applyStopSequences(const std::string& text,
                                                const GenerationParams& params) const {
    std::string result = text;
    for (const auto& stop_seq : params.stop_sequences) {
        size_t pos = result.find(stop_seq);
        if (pos != std::string::npos) {
            result = result.substr(0, pos);
            break;
        }
    }
    return result;
}

std::string InferenceEngine::extractDeltaPromptFromLastUser(const std::string& full_prompt) const {
    static constexpr const char* kUserTurnMarker = "<|im_start|>user";
    const size_t pos = full_prompt.rfind(kUserTurnMarker);
    if (pos == std::string::npos) {
        throw std::runtime_error(
            std::string("Multi-turn: could not find latest user turn marker '") + kUserTurnMarker +
            "' in full prompt");
    }
    return full_prompt.substr(pos);
}

InferenceEngine::ChatSession& InferenceEngine::getOrCreateChatSession(const std::string& conversation_id) {
    auto it = chat_sessions_.find(conversation_id);
    if (it != chat_sessions_.end()) {
        return it->second;
    }

    ChatSession session;
    session.gen_params = OgaGeneratorParams::Create(*model_);
    session.generator = OgaGenerator::Create(*model_, *session.gen_params);
    session.turn_count = 0;

    auto [inserted_it, inserted] = chat_sessions_.emplace(conversation_id, std::move(session));
    if (!inserted) {
        throw std::runtime_error("Failed to create chat session: " + conversation_id);
    }

    std::cout << "[InferenceEngine] Created multi-turn session: " << conversation_id << std::endl;
    return inserted_it->second;
}

void InferenceEngine::resetMultiTurnSession(const std::string& conversation_id) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    resetMultiTurnSessionLocked(conversation_id);
}

void InferenceEngine::resetMultiTurnSessionLocked(const std::string& conversation_id) {
    auto it = chat_sessions_.find(conversation_id);
    if (it == chat_sessions_.end()) {
        std::cout << "[InferenceEngine] Reset multi-turn session: " << conversation_id
                  << " (no existing session, no-op)" << std::endl;
        return;
    }

    ChatSession& session = it->second;
    try {
        session.generator->RewindTo(0);
        session.turn_count = 0;
        std::cout << "[InferenceEngine] Reset multi-turn session: " << conversation_id
                  << " (RewindTo(0), turn_count=0)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] RewindTo(0) failed for session " << conversation_id
                  << ", recreating generator: " << e.what() << std::endl;
        chat_sessions_.erase(it);
    }
}

std::string InferenceEngine::completeMultiTurn(const std::string& conversation_id,
                                               const std::string& messages_json,
                                               const GenerationParams& params,
                                               CompletionTimingData* out_timing) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    try {
        const std::string full_prompt = applyChatTemplate(messages_json, "", true);
        ChatSession& session = getOrCreateChatSession(conversation_id);
        const bool is_first_turn = (session.turn_count == 0);
        const std::string delta_prompt = is_first_turn
            ? full_prompt
            : extractDeltaPromptFromLastUser(full_prompt);
        const std::string text_to_append = delta_prompt;

        if (!is_first_turn && text_to_append.empty()) {
            throw std::runtime_error("Multi-turn request produced an empty delta prompt");
        }

        const int append_token_count = countTokens(text_to_append);
        const int full_token_count = countTokens(full_prompt);

        logMultiTurnAppendPlan("text", conversation_id, session.turn_count + 1, is_first_turn,
                               full_prompt, text_to_append, full_token_count, append_token_count,
                               "AppendTokens");

        writeAppendDebugFile(is_first_turn ? "APPEND_FULL" : "APPEND_DELTA",
                             conversation_id,
                             session.turn_count + 1,
                             text_to_append);

        const size_t seq_before = session.generator->GetSequenceCount(0);
        appendPromptText(*session.generator, text_to_append);
        const size_t seq_after_append = session.generator->GetSequenceCount(0);
        logMultiTurnAfterAppend("text", seq_before, seq_after_append, session.generator->IsDone());

        int total_max_length = static_cast<int>(session.generator->GetSequenceCount(0)) + params.max_length;
        if (model_context_length_ > 0 && total_max_length > model_context_length_) {
            total_max_length = model_context_length_;
        }
        configureGeneratorParams(*session.gen_params, params, total_max_length);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto first_token_time = start_time;
        bool first_token_received = false;

        while (!session.generator->IsDone()) {
            session.generator->GenerateNextToken();
            if (!first_token_received) {
                first_token_time = std::chrono::high_resolution_clock::now();
                first_token_received = true;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        const int32_t* output_ptr = session.generator->GetSequenceData(0);
        const size_t output_count = session.generator->GetSequenceCount(0);
        const int generated_token_count = (output_count > seq_before)
            ? static_cast<int>(output_count - seq_before)
            : 0;

        std::string result;
        if (output_count > seq_before) {
            auto decoded = tokenizer_->Decode(output_ptr + seq_before, output_count - seq_before);
            result = applyStopSequences(std::string(decoded), params);
        }

        session.turn_count++;

        logMultiTurnCompleted("text", session.turn_count, seq_before, output_count,
                              generated_token_count, -1);

        if (out_timing != nullptr) {
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            auto ttft_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
            const double ttft_seconds = ttft_duration.count() / 1000.0;
            const double total_time_ms = static_cast<double>(total_duration.count());
            const double decode_time_seconds = (total_duration.count() - ttft_duration.count()) / 1000.0;
            double tps = 0.0;
            if (generated_token_count > 1 && decode_time_seconds > 0) {
                tps = (generated_token_count - 1) / decode_time_seconds;
            } else if (generated_token_count == 1 && total_time_ms > 0) {
                tps = 1.0 / (total_time_ms / 1000.0);
            }

            out_timing->token_count = generated_token_count;
            out_timing->ttft_seconds = ttft_seconds;
            out_timing->tps = tps;
            out_timing->total_time_ms = total_time_ms;
        }

        std::cout << "[InferenceEngine] Multi-turn completed turn=" << session.turn_count
                  << " generated=" << generated_token_count
                  << " token_count=" << session.generator->TokenCount() << std::endl;

        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error("Multi-turn inference failed: " + std::string(e.what()));
    }
}

void InferenceEngine::streamMultiTurn(const std::string& conversation_id,
                                      const std::string& messages_json,
                                      const GenerationParams& params,
                                      StreamCallback callback) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    try {
        const std::string full_prompt = applyChatTemplate(messages_json, "", true);
        ChatSession& session = getOrCreateChatSession(conversation_id);
        const bool is_first_turn = (session.turn_count == 0);
        const std::string text_to_append = is_first_turn
            ? full_prompt
            : extractDeltaPromptFromLastUser(full_prompt);

        if (!is_first_turn && text_to_append.empty()) {
            throw std::runtime_error("Multi-turn request produced an empty delta prompt");
        }

        const int append_token_count = countTokens(text_to_append);
        const int full_token_count = countTokens(full_prompt);

        logMultiTurnAppendPlan("text-stream", conversation_id, session.turn_count + 1, is_first_turn,
                               full_prompt, text_to_append, full_token_count, append_token_count,
                               "AppendTokens");

        writeAppendDebugFile(is_first_turn ? "APPEND_FULL" : "APPEND_DELTA",
                             conversation_id,
                             session.turn_count + 1,
                             text_to_append);

        const size_t seq_before = session.generator->GetSequenceCount(0);
        appendPromptText(*session.generator, text_to_append);
        const size_t seq_after_append = session.generator->GetSequenceCount(0);
        logMultiTurnAfterAppend("text-stream", seq_before, seq_after_append, session.generator->IsDone());

        int total_max_length = static_cast<int>(session.generator->GetSequenceCount(0)) + params.max_length;
        if (model_context_length_ > 0 && total_max_length > model_context_length_) {
            total_max_length = model_context_length_;
        }
        configureGeneratorParams(*session.gen_params, params, total_max_length);

        auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer_);
        std::string accumulated_output;
        bool client_disconnected = false;

        while (!session.generator->IsDone() && !client_disconnected) {
            session.generator->GenerateNextToken();

            const int32_t* all_tokens = session.generator->GetSequenceData(0);
            const size_t num_tokens = session.generator->GetSequenceCount(0);
            const int32_t new_token = all_tokens[num_tokens - 1];

            const char* decoded = tokenizer_stream->Decode(new_token);
            if (decoded && decoded[0] != '\0') {
                std::string token_str(decoded);

                bool should_stop = false;
                for (const auto& stop_seq : params.stop_sequences) {
                    std::string temp_output = accumulated_output + token_str;
                    if (temp_output.find(stop_seq) != std::string::npos) {
                        should_stop = true;
                        break;
                    }
                }
                if (should_stop) {
                    break;
                }

                accumulated_output += token_str;
                const bool is_final = session.generator->IsDone();
                if (!callback(token_str, is_final)) {
                    client_disconnected = true;
                    break;
                }
            }
        }

        accumulated_output = applyStopSequences(accumulated_output, params);
        session.turn_count++;

        const size_t seq_after = session.generator->GetSequenceCount(0);
        const int generated_token_count = (seq_after > seq_before)
            ? static_cast<int>(seq_after - seq_before)
            : 0;
        logMultiTurnCompleted("text-stream", session.turn_count, seq_before, seq_after,
                              generated_token_count, -1);

        std::cout << "[InferenceEngine] Multi-turn stream completed turn=" << session.turn_count
                  << " token_count=" << session.generator->TokenCount() << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Multi-turn streaming failed: " + std::string(e.what()));
    }
}

void InferenceEngine::detectMultimodal() {
    // A model is multimodal if genai_config.json declares a vision (or speech)
    // pipeline, or a processor_config.json is present alongside the model.
    bool looks_multimodal = false;
    try {
        std::string config_path = model_path_ + "/genai_config.json";
        if (fs::exists(config_path)) {
            std::ifstream file(config_path);
            json config = json::parse(file);
            const auto& model = config["model"];
            if (model.contains("vision") || model.contains("speech") ||
                model.contains("image_token_id")) {
                looks_multimodal = true;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] multimodal detection failed: " << e.what() << std::endl;
    }
    if (!looks_multimodal && fs::exists(model_path_ + "/processor_config.json")) {
        looks_multimodal = true;
    }

    if (!looks_multimodal) {
        return;
    }

    try {
        processor_ = OgaMultiModalProcessor::Create(*model_);
        is_multimodal_ = true;
        std::cout << "[InferenceEngine] Multimodal (vision) pipeline initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to create multimodal processor: " << e.what()
                  << " (continuing as text-only)" << std::endl;
        is_multimodal_ = false;
    }
}

std::string InferenceEngine::applyChatTemplateRaw(const std::string& messages_json,
                                                  const std::string& tools_json) {
    const char* template_str = chat_template_.empty() ? nullptr : chat_template_.c_str();
    const char* tools_str = tools_json.empty() ? nullptr : tools_json.c_str();
    auto result = tokenizer_->ApplyChatTemplate(template_str, messages_json.c_str(), tools_str, true);
    return std::string(result);
}

namespace {
// Configure search options on generator params, shared by the multimodal paths.
void apply_search_options(OgaGeneratorParams& gen_params, const GenerationParams& params,
                          int total_max_length) {
    gen_params.SetSearchOption("max_length", total_max_length);
    gen_params.SetSearchOption("temperature", params.temperature);
    gen_params.SetSearchOption("top_p", params.top_p);
    gen_params.SetSearchOption("top_k", static_cast<double>(params.top_k));
    gen_params.SetSearchOption("repetition_penalty", params.repetition_penalty);
    gen_params.SetSearchOptionBool("do_sample", params.do_sample);
    gen_params.SetSearchOption("random_seed", 1.0);
}
}  // namespace

namespace {

std::unique_ptr<OgaImages> loadOgaImages(const std::vector<std::string>& images) {
    if (images.empty()) {
        return nullptr;
    }
    std::vector<const void*> data_ptrs;
    std::vector<size_t> data_sizes;
    data_ptrs.reserve(images.size());
    data_sizes.reserve(images.size());
    for (const auto& img : images) {
        data_ptrs.push_back(img.data());
        data_sizes.push_back(img.size());
    }
    return OgaImages::Load(data_ptrs.data(), data_sizes.data(), data_ptrs.size());
}

}  // namespace

std::string InferenceEngine::completeMultiTurnMultimodal(const std::string& conversation_id,
                                                         const std::string& messages_json,
                                                         const std::vector<std::string>& new_turn_images,
                                                         const std::vector<std::string>& all_images,
                                                         const std::string& tools_json,
                                                         const GenerationParams& params,
                                                         CompletionTimingData* out_timing) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    if (!processor_) {
        throw std::runtime_error("Multimodal multi-turn requested but model is not multimodal");
    }

    try {
        const std::string full_prompt = applyChatTemplateRaw(messages_json, tools_json);
        bool is_first_turn = true;
        if (auto it = chat_sessions_.find(conversation_id); it != chat_sessions_.end()) {
            is_first_turn = (it->second.turn_count == 0);
        }

        if (!is_first_turn && !new_turn_images.empty()) {
            std::cout << "[InferenceEngine] Multimodal multi-turn: new images on follow-up turn, "
                      << "resetting session and reprocessing full prompt" << std::endl;
            chat_sessions_.erase(conversation_id);
            is_first_turn = true;
        }

        ChatSession& active_session = getOrCreateChatSession(conversation_id);
        const std::string text_to_append = is_first_turn
            ? full_prompt
            : extractDeltaPromptFromLastUser(full_prompt);

        if (text_to_append.empty()) {
            throw std::runtime_error("Multimodal multi-turn request produced an empty prompt segment");
        }

        const std::vector<std::string>& images_for_process =
            is_first_turn ? all_images : new_turn_images;

        const int append_token_count = countTokens(text_to_append);
        const int full_token_count = countTokens(full_prompt);
        const char* input_mode =
            (is_first_turn && !images_for_process.empty()) ? "SetInputs" : "AppendTokens";

        logMultiTurnAppendPlan("vlm", conversation_id, active_session.turn_count + 1, is_first_turn,
                               full_prompt, text_to_append, full_token_count, append_token_count,
                               input_mode);

        writeAppendDebugFile(is_first_turn ? "APPEND_FULL" : "APPEND_DELTA",
                             conversation_id,
                             active_session.turn_count + 1,
                             text_to_append);

        const size_t seq_before = active_session.generator->GetSequenceCount(0);

        if (is_first_turn && !images_for_process.empty()) {
            auto oga_images = loadOgaImages(images_for_process);
            auto inputs = processor_->ProcessImages(text_to_append.c_str(), oga_images.get());
            active_session.generator->SetInputs(*inputs);
        } else {
            appendPromptText(*active_session.generator, text_to_append);
        }

        const size_t seq_after_append = active_session.generator->GetSequenceCount(0);
        logMultiTurnAfterAppend("vlm", seq_before, seq_after_append, active_session.generator->IsDone());

        int total_max_length = static_cast<int>(active_session.generator->GetSequenceCount(0)) + params.max_length;
        if (model_context_length_ > 0 && total_max_length > model_context_length_) {
            total_max_length = model_context_length_;
        }
        configureGeneratorParams(*active_session.gen_params, params, total_max_length);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto first_token_time = start_time;
        bool first_token_received = false;

        while (!active_session.generator->IsDone()) {
            active_session.generator->GenerateNextToken();
            if (!first_token_received) {
                first_token_time = std::chrono::high_resolution_clock::now();
                first_token_received = true;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        const int32_t* output_ptr = active_session.generator->GetSequenceData(0);
        const size_t output_count = active_session.generator->GetSequenceCount(0);
        const int generated_token_count = (output_count > seq_before)
            ? static_cast<int>(output_count - seq_before)
            : 0;

        std::string result;
        if (output_count > seq_before) {
            auto decoded = processor_->Decode(output_ptr + seq_before, output_count - seq_before);
            result = applyStopSequences(std::string(decoded), params);
        }

        active_session.turn_count++;

        logMultiTurnCompleted("vlm", active_session.turn_count, seq_before, output_count,
                              generated_token_count, -1);

        if (out_timing != nullptr) {
            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            auto ttft_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
            const double ttft_seconds = ttft_duration.count() / 1000.0;
            const double total_time_ms = static_cast<double>(total_duration.count());
            const double decode_time_seconds = (total_duration.count() - ttft_duration.count()) / 1000.0;
            double tps = 0.0;
            if (generated_token_count > 1 && decode_time_seconds > 0) {
                tps = (generated_token_count - 1) / decode_time_seconds;
            } else if (generated_token_count == 1 && total_time_ms > 0) {
                tps = 1.0 / (total_time_ms / 1000.0);
            }

            out_timing->token_count = generated_token_count;
            out_timing->ttft_seconds = ttft_seconds;
            out_timing->tps = tps;
            out_timing->total_time_ms = total_time_ms;
        }

        std::cout << "[InferenceEngine] Multimodal multi-turn completed turn=" << active_session.turn_count
                  << " generated=" << generated_token_count << std::endl;
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error("Multimodal multi-turn inference failed: " + std::string(e.what()));
    }
}

void InferenceEngine::streamMultiTurnMultimodal(const std::string& conversation_id,
                                                const std::string& messages_json,
                                                const std::vector<std::string>& new_turn_images,
                                                const std::vector<std::string>& all_images,
                                                const std::string& tools_json,
                                                const GenerationParams& params,
                                                StreamCallback callback) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    if (!processor_) {
        throw std::runtime_error("Multimodal multi-turn requested but model is not multimodal");
    }

    try {
        const std::string full_prompt = applyChatTemplateRaw(messages_json, tools_json);
        bool is_first_turn = true;
        if (auto it = chat_sessions_.find(conversation_id); it != chat_sessions_.end()) {
            is_first_turn = (it->second.turn_count == 0);
        }

        if (!is_first_turn && !new_turn_images.empty()) {
            std::cout << "[InferenceEngine] Multimodal multi-turn stream: new images on follow-up turn, "
                      << "resetting session and reprocessing full prompt" << std::endl;
            chat_sessions_.erase(conversation_id);
            is_first_turn = true;
        }

        ChatSession& session = getOrCreateChatSession(conversation_id);

        const std::string text_to_append = is_first_turn
            ? full_prompt
            : extractDeltaPromptFromLastUser(full_prompt);

        if (text_to_append.empty()) {
            throw std::runtime_error("Multimodal multi-turn request produced an empty prompt segment");
        }

        const std::vector<std::string>& images_for_process =
            is_first_turn ? all_images : new_turn_images;

        const int append_token_count = countTokens(text_to_append);
        const int full_token_count = countTokens(full_prompt);
        const char* input_mode =
            (is_first_turn && !images_for_process.empty()) ? "SetInputs" : "AppendTokens";

        logMultiTurnAppendPlan("vlm-stream", conversation_id, session.turn_count + 1, is_first_turn,
                               full_prompt, text_to_append, full_token_count, append_token_count,
                               input_mode);
        std::cout << "[InferenceEngine][MultiTurnDebug] vlm-stream new_turn_images="
                  << new_turn_images.size() << " process_images=" << images_for_process.size()
                  << std::endl;

        writeAppendDebugFile(is_first_turn ? "APPEND_FULL" : "APPEND_DELTA",
                             conversation_id,
                             session.turn_count + 1,
                             text_to_append);

        const size_t seq_before = session.generator->GetSequenceCount(0);

        if (is_first_turn && !images_for_process.empty()) {
            auto oga_images = loadOgaImages(images_for_process);
            auto inputs = processor_->ProcessImages(text_to_append.c_str(), oga_images.get());
            session.generator->SetInputs(*inputs);
        } else {
            appendPromptText(*session.generator, text_to_append);
        }

        const size_t seq_after_append = session.generator->GetSequenceCount(0);
        logMultiTurnAfterAppend("vlm-stream", seq_before, seq_after_append, session.generator->IsDone());

        int total_max_length = static_cast<int>(session.generator->GetSequenceCount(0)) + params.max_length;
        if (model_context_length_ > 0 && total_max_length > model_context_length_) {
            total_max_length = model_context_length_;
        }
        configureGeneratorParams(*session.gen_params, params, total_max_length);

        auto tokenizer_stream = OgaTokenizerStream::Create(*processor_);
        std::string accumulated_output;
        bool client_disconnected = false;

        while (!session.generator->IsDone() && !client_disconnected) {
            session.generator->GenerateNextToken();

            const int32_t* all_tokens = session.generator->GetSequenceData(0);
            const size_t num_tokens = session.generator->GetSequenceCount(0);
            const int32_t new_token = all_tokens[num_tokens - 1];

            const char* decoded = tokenizer_stream->Decode(new_token);
            if (decoded && decoded[0] != '\0') {
                std::string token_str(decoded);

                bool should_stop = false;
                for (const auto& stop_seq : params.stop_sequences) {
                    std::string temp_output = accumulated_output + token_str;
                    if (temp_output.find(stop_seq) != std::string::npos) {
                        should_stop = true;
                        break;
                    }
                }
                if (should_stop) {
                    break;
                }

                accumulated_output += token_str;
                const bool is_final = session.generator->IsDone();
                if (!callback(token_str, is_final)) {
                    client_disconnected = true;
                    break;
                }
            }
        }

        accumulated_output = applyStopSequences(accumulated_output, params);
        session.turn_count++;

        const size_t seq_after = session.generator->GetSequenceCount(0);
        const int generated_token_count = (seq_after > seq_before)
            ? static_cast<int>(seq_after - seq_before)
            : 0;
        logMultiTurnCompleted("vlm-stream", session.turn_count, seq_before, seq_after,
                              generated_token_count, -1);

        std::cout << "[InferenceEngine] Multimodal multi-turn stream completed turn="
                  << session.turn_count << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Multimodal multi-turn streaming failed: " + std::string(e.what()));
    }
}

std::string InferenceEngine::completeMultimodal(const std::string& prompt,
                                                const std::vector<std::string>& images,
                                                const GenerationParams& params,
                                                CompletionTimingData* out_timing) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    if (!processor_) {
        throw std::runtime_error("Multimodal completion requested but model is not multimodal");
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto first_token_time = start_time;
    bool first_token_received = false;

    // Load images from in-memory buffers.
    std::vector<const void*> data_ptrs;
    std::vector<size_t> data_sizes;
    for (const auto& img : images) {
        data_ptrs.push_back(img.data());
        data_sizes.push_back(img.size());
    }
    std::unique_ptr<OgaImages> oga_images;
    if (!images.empty()) {
        oga_images = OgaImages::Load(data_ptrs.data(), data_sizes.data(), data_ptrs.size());
    }

    auto inputs = processor_->ProcessImages(prompt.c_str(), oga_images.get());

    int input_len = 0;
    try {
        auto ids = inputs->Get("input_ids");
        auto shape = ids->Shape();
        if (!shape.empty()) input_len = static_cast<int>(shape.back());
    } catch (...) {}

    int total_max_length = input_len + params.max_length;
    if (model_context_length_ > 0 && total_max_length > model_context_length_) {
        total_max_length = model_context_length_;
    }
    auto gen_params = OgaGeneratorParams::Create(*model_);
    apply_search_options(*gen_params, params, total_max_length);

    auto generator = OgaGenerator::Create(*model_, *gen_params);
    generator->SetInputs(*inputs);

    auto stream = OgaTokenizerStream::Create(*processor_);
    std::string result;
    int generated = 0;
    while (!generator->IsDone() && generated < params.max_length) {
        generator->GenerateNextToken();
        if (!first_token_received) {
            first_token_time = std::chrono::high_resolution_clock::now();
            first_token_received = true;
        }
        const int32_t* toks = generator->GetSequenceData(0);
        size_t n = generator->GetSequenceCount(0);
        const char* decoded = stream->Decode(toks[n - 1]);
        if (decoded && decoded[0] != '\0') result += decoded;
        generated++;

        bool stop_hit = false;
        for (const auto& stop_seq : params.stop_sequences) {
            size_t pos = result.find(stop_seq);
            if (pos != std::string::npos) { result = result.substr(0, pos); stop_hit = true; break; }
        }
        if (stop_hit) break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    if (out_timing) {
        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto ttft = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        double decode_s = (total.count() - ttft.count()) / 1000.0;
        out_timing->token_count = generated;
        out_timing->ttft_seconds = ttft.count() / 1000.0;
        out_timing->tps = (generated > 1 && decode_s > 0) ? (generated - 1) / decode_s : 0.0;
        out_timing->total_time_ms = static_cast<double>(total.count());
    }
    std::cout << "[InferenceEngine] Generated " << generated << " tokens (multimodal)" << std::endl;
    return result;
}

void InferenceEngine::streamCompleteMultimodal(const std::string& prompt,
                                               const std::vector<std::string>& images,
                                               const GenerationParams& params,
                                               StreamCallback callback) {
    std::lock_guard<std::mutex> lock(inference_mutex_);
    if (!processor_) {
        throw std::runtime_error("Multimodal completion requested but model is not multimodal");
    }

    std::vector<const void*> data_ptrs;
    std::vector<size_t> data_sizes;
    for (const auto& img : images) {
        data_ptrs.push_back(img.data());
        data_sizes.push_back(img.size());
    }
    std::unique_ptr<OgaImages> oga_images;
    if (!images.empty()) {
        oga_images = OgaImages::Load(data_ptrs.data(), data_sizes.data(), data_ptrs.size());
    }

    auto inputs = processor_->ProcessImages(prompt.c_str(), oga_images.get());

    int input_len = 0;
    try {
        auto ids = inputs->Get("input_ids");
        auto shape = ids->Shape();
        if (!shape.empty()) input_len = static_cast<int>(shape.back());
    } catch (...) {}

    int total_max_length = input_len + params.max_length;
    if (model_context_length_ > 0 && total_max_length > model_context_length_) {
        total_max_length = model_context_length_;
    }
    auto gen_params = OgaGeneratorParams::Create(*model_);
    apply_search_options(*gen_params, params, total_max_length);

    auto generator = OgaGenerator::Create(*model_, *gen_params);
    generator->SetInputs(*inputs);

    auto stream = OgaTokenizerStream::Create(*processor_);
    std::string accumulated;
    int generated = 0;
    bool client_disconnected = false;
    while (!generator->IsDone() && generated < params.max_length && !client_disconnected) {
        generator->GenerateNextToken();
        const int32_t* toks = generator->GetSequenceData(0);
        size_t n = generator->GetSequenceCount(0);
        const char* decoded = stream->Decode(toks[n - 1]);
        generated++;
        if (decoded && decoded[0] != '\0') {
            std::string token_str(decoded);
            bool should_stop = false;
            for (const auto& stop_seq : params.stop_sequences) {
                if ((accumulated + token_str).find(stop_seq) != std::string::npos) { should_stop = true; break; }
            }
            if (should_stop) break;
            accumulated += token_str;
            bool is_final = generator->IsDone() || generated >= params.max_length;
            if (!callback(token_str, is_final)) {
                client_disconnected = true;
                break;
            }
        }
    }
    std::cout << "[InferenceEngine] Generated " << generated << " tokens (multimodal stream)" << std::endl;
}

} // namespace ryzenai

