#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

// Read MTP-head draft tokens from the target context's most recent decode.
//
// Uses llama_get_mtp_logits + llama_get_mtp_n_drafts + llama_get_mtp_n_vocab,
// which come from the model's MTP head output (chained rollout when built
// with n_draft_rollout > 1). Returns up to k_max draft tokens, stopping
// early on an EOG prediction. Returns an empty vector if the model has no
// MTP head or the buffer is unavailable.
//
// Consumers: both the standard speculative framework
// (common_speculative_state_mtp::draft()) and the inline two-phase MTP
// producer in tools/server/server-context.cpp use this helper. Extracted
// to avoid drift between the two code paths and to put all chained-rollout
// awareness in one place.
llama_tokens common_mtp_read_drafts(llama_context * ctx_tgt, int k_max);
