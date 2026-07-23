#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>

// Forward declarations for ONNX Runtime GenAI
struct OgaModel;
struct OgaTokenizer;
struct OgaGeneratorParams;
struct OgaGenerator;
struct OgaSequences;
struct OgaMultiModalProcessor;

namespace ryzenai {

// Timing data returned from completion
struct CompletionTimingData {
    int token_count = 0;           // Number of generated tokens
    double ttft_seconds = 0.0;     // Time to first token in seconds
    double tps = 0.0;              // Tokens per second (decode speed)
    double total_time_ms = 0.0;    // Total completion time in milliseconds
};

class InferenceEngine {
public:
    // ctx_size: requested context window from --ctx-size. <=0 means "use the
    // model's native context_length from genai_config.json".
    InferenceEngine(const std::string& model_path, int ctx_size = 0);
    ~InferenceEngine();
    
    // Synchronous completion
    // Returns generated text. If out_timing is provided, stores timing data.
    std::string complete(const std::string& prompt, const GenerationParams& params, CompletionTimingData* out_timing = nullptr);
    
    // Streaming completion
    void streamComplete(const std::string& prompt, 
                       const GenerationParams& params,
                       StreamCallback callback);
    
    // Apply chat template to messages
    std::string applyChatTemplate(const std::string& messages_json, const std::string& tools_json = "");

    // Apply the model's chat template strictly via OGA (jinja), without the
    // text-only manual fallbacks. Required for multimodal models whose template
    // expands image placeholders (e.g. <|vision_start|><|image_pad|><|vision_end|>).
    std::string applyChatTemplateRaw(const std::string& messages_json, const std::string& tools_json = "");

    // True when the model carries a vision/embedding pipeline (VLM).
    bool isMultimodal() const { return is_multimodal_; }

    // Multimodal completion: prompt is already chat-templated; images are raw
    // encoded image bytes (e.g. JPEG/PNG), one entry per image, in prompt order.
    std::string completeMultimodal(const std::string& prompt,
                                   const std::vector<std::string>& images,
                                   const GenerationParams& params,
                                   CompletionTimingData* out_timing = nullptr);

    void streamCompleteMultimodal(const std::string& prompt,
                                  const std::vector<std::string>& images,
                                  const GenerationParams& params,
                                  StreamCallback callback);

    // Getters
    std::string getModelName() const { return model_name_; }
    std::string getExecutionMode() const { return execution_mode_; }
    int getMaxPromptLength() const { return max_prompt_length_; }
    std::string getRyzenAIVersion() const { return ryzenai_version_; }
    
    // Get default generation params from genai_config.json (if available)
    GenerationParams getDefaultParams() const;
    
    // Token counting
    int countTokens(const std::string& text);
    
private:
    void loadModel();
    void detectMultimodal();
    void setupExecutionProvider();
    void loadRaiConfig();
    std::string detectRyzenAIVersion();
    std::string detectExecutionMode();
    std::string resolveModelPath(const std::string& path);
    std::vector<int32_t> truncatePrompt(const std::vector<int32_t>& input_ids);
    bool validateModelDirectory(const std::string& path);
    
    std::unique_ptr<OgaModel> model_;
    std::unique_ptr<OgaTokenizer> tokenizer_;
    std::unique_ptr<OgaMultiModalProcessor> processor_;  // only for multimodal models
    bool is_multimodal_ = false;

    std::string model_path_;
    std::string model_name_;
    std::string execution_mode_;  // "npu", "hybrid", or "cpu"
    std::string ryzenai_version_;
    std::string chat_template_;  // Chat template from tokenizer_config.json
    int max_prompt_length_ = 2048;  // Default, overridden by rai_config.json / context window
    int ctx_size_ = 0;              // Requested context window (<=0 = use model native)
    int model_context_length_ = 0;  // Native context_length from genai_config.json
    bool rai_max_prompt_set_ = false;  // True if rai_config.json pinned max_prompt_length
    
    // Default generation params from genai_config.json search section
    GenerationParams default_params_;
    bool has_search_config_ = false;
    
    std::mutex inference_mutex_;  // Protect inference operations
};

} // namespace ryzenai

