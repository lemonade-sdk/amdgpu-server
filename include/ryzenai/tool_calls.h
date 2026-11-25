#pragma once

#include <string>
#include <vector>
#include <json.hpp>
#include <regex>

namespace ryzenai {

using json = nlohmann::json;

// Extracted tool call structure
struct ToolCall {
    std::string name;
    json arguments;
};

// Escape a raw string so it is safe to embed in a JSON string literal.
// Handles quotes, backslashes, the named control escapes, and any remaining
// character below 0x20 as \u00XX.
std::string escapeJsonString(const std::string& s);

// Extract tool calls from generated text (Qwen format: <tool_call>...</tool_call>)
// Returns: pair of (extracted_tool_calls, cleaned_text_without_tool_calls)
std::pair<std::vector<ToolCall>, std::string> extractToolCalls(const std::string& text);

// Streaming counterpart of extractToolCalls(). Content is pushed in as it is
// generated, and text that is, or may still become, tool-call markup is held
// back so the raw markup is not streamed to the client as content.
//
// Everything from the first tool-call marker on is held back until finish().
// The caller runs extractToolCalls() on that text: the calls it finds go out as
// tool_calls, and what is left (prose, or markup that did not parse) goes out
// as content, so no text is lost.
class ToolCallStreamFilter {
public:
    // Add the next piece of content. Returns the part that can be sent now.
    std::string push(const std::string& text);

    // End of the stream. Returns the text that was held back: everything from
    // the first tool-call marker on, or a trailing partial marker.
    std::string finish();

    // True once a tool-call marker has been seen. From then on push() returns
    // nothing, so nothing is written to the client until finish().
    bool holding() const { return holding_; }

private:
    std::string buffer_;  // held-back text, or a tail that may still become a marker
    bool holding_ = false;
};

// Format tool calls in OpenAI API format
json formatToolCallsForOpenAI(const std::vector<ToolCall>& tool_calls);

// Build chat.completion.chunk JSON payloads (without the "data: " prefix) for
// streaming tool_calls deltas. Uses the same OpenAI shape as formatToolCallsForOpenAI.
std::vector<std::string> buildToolCallStreamChunks(const json& openai_tool_calls,
                                                   const std::string& model_id);

} // namespace ryzenai

