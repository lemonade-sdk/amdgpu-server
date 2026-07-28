#include "ryzenai/types.h"
#include <sstream>
#include <iostream>

namespace ryzenai {

CompletionRequest CompletionRequest::fromJSON(const json& j) {
    CompletionRequest req;
    
    if (j.contains("prompt") && j["prompt"].is_string()) {
        req.prompt = j["prompt"];
    }
    
    // Support both max_tokens (deprecated) and max_completion_tokens (newer OpenAI API)
    // max_completion_tokens takes precedence if both are provided
    if (j.contains("max_completion_tokens")) {
        req.max_tokens = j["max_completion_tokens"];
    } else if (j.contains("max_tokens")) {
        req.max_tokens = j["max_tokens"];
    }
    
    if (j.contains("temperature")) {
        req.temperature = j["temperature"];
    }
    
    if (j.contains("top_p")) {
        req.top_p = j["top_p"];
    }
    
    if (j.contains("top_k")) {
        req.top_k = j["top_k"];
    }
    
    if (j.contains("repeat_penalty")) {
        req.repeat_penalty = j["repeat_penalty"];
    } else if (j.contains("repetition_penalty")) {
        req.repeat_penalty = j["repetition_penalty"];
    } else if (j.contains("frequency_penalty")) {
        // OpenAI uses frequency_penalty, we map it to repeat_penalty
        req.repeat_penalty = 1.0f + j["frequency_penalty"].get<float>();
    }
    
    if (j.contains("stream")) {
        req.stream = j["stream"];
    }
    
    if (j.contains("echo")) {
        req.echo = j["echo"];
    }
    
    if (j.contains("stop")) {
        if (j["stop"].is_string()) {
            req.stop.push_back(j["stop"]);
        } else if (j["stop"].is_array()) {
            for (const auto& s : j["stop"]) {
                req.stop.push_back(s);
            }
        }
    }
    
    return req;
}

ChatCompletionRequest ChatCompletionRequest::fromJSON(const json& j) {
    ChatCompletionRequest req;
    
    if (j.contains("messages") && j["messages"].is_array()) {
        for (const auto& msg : j["messages"]) {
            ChatMessage message;
            message.role = msg.value("role", "user");
            // Content may be a plain string or an OpenAI structured array
            // (text + image_url parts). For the array form, concatenate the
            // text parts here; images are handled separately via the raw request.
            if (msg.contains("content")) {
                const auto& content = msg["content"];
                if (content.is_string()) {
                    message.content = content.get<std::string>();
                } else if (content.is_array()) {
                    std::string text;
                    for (const auto& item : content) {
                        if (item.is_object() && item.value("type", "") == "text" &&
                            item.contains("text") && item["text"].is_string()) {
                            if (!text.empty()) text += "\n";
                            text += item["text"].get<std::string>();
                        }
                    }
                    message.content = text;
                }
            }
            req.messages.push_back(message);
        }
    }
    
    // Support both max_tokens (deprecated) and max_completion_tokens (newer OpenAI API)
    // max_completion_tokens takes precedence if both are provided
    if (j.contains("max_completion_tokens")) {
        req.max_tokens = j["max_completion_tokens"];
        std::cout << "[ChatCompletionRequest] Parsed max_completion_tokens=" << req.max_tokens << std::endl;
    } else if (j.contains("max_tokens")) {
        req.max_tokens = j["max_tokens"];
        std::cout << "[ChatCompletionRequest] Parsed max_tokens=" << req.max_tokens << std::endl;
    } else {
        std::cout << "[ChatCompletionRequest] No max_tokens specified, using default=" << req.max_tokens << std::endl;
    }
    
    if (j.contains("temperature")) {
        req.temperature = j["temperature"];
    }
    
    if (j.contains("top_p")) {
        req.top_p = j["top_p"];
    }
    
    if (j.contains("top_k")) {
        req.top_k = j["top_k"];
    }
    
    if (j.contains("repeat_penalty")) {
        req.repeat_penalty = j["repeat_penalty"];
    } else if (j.contains("repetition_penalty")) {
        req.repeat_penalty = j["repetition_penalty"];
    } else if (j.contains("frequency_penalty")) {
        req.repeat_penalty = 1.0f + j["frequency_penalty"].get<float>();
    }
    
    if (j.contains("stream")) {
        req.stream = j["stream"];
    }
    
    if (j.contains("stop")) {
        if (j["stop"].is_string()) {
            req.stop.push_back(j["stop"]);
        } else if (j["stop"].is_array()) {
            for (const auto& s : j["stop"]) {
                req.stop.push_back(s);
            }
        }
    }
    
    if (j.contains("tools")) {
        req.tools = j["tools"];
    }

    if (j.contains("conversation_id") && j["conversation_id"].is_string()) {
        req.conversation_id = j["conversation_id"];
    }
    
    return req;
}

std::string ChatCompletionRequest::toPrompt() const {
    std::ostringstream prompt;
    
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        
        // Simple chat template formatting
        if (msg.role == "system") {
            prompt << "System: " << msg.content << "\n\n";
        } else if (msg.role == "user") {
            prompt << "User: " << msg.content << "\n\n";
        } else if (msg.role == "assistant") {
            prompt << "Assistant: " << msg.content << "\n\n";
        }
    }
    
    // Add final "Assistant: " prompt for the model to complete
    prompt << "Assistant: ";
    
    return prompt.str();
}

} // namespace ryzenai

