#include "ggml.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static llama_token sample_dist(llama_sampler * sampler, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < (llama_token) logits.size(); ++token_id) {
        cur.push_back({ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0);
    GGML_ASSERT((size_t) cur_p.selected < cur_p.size);
    return cur_p.data[cur_p.selected].id;
}

static void test_dist_singleton_rng() {
    llama_sampler * singleton = llama_sampler_init_dist(4242);
    llama_sampler * control   = llama_sampler_init_dist(4242);

    sample_dist(singleton, { 0.0f });
    sample_dist(control,   { 0.0f, 0.0f });

    const std::vector<float> logits(256, 0.0f);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(sample_dist(singleton, logits) == sample_dist(control, logits));
    }

    llama_sampler_free(singleton);
    llama_sampler_free(control);
}

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static std::vector<float> ngram_run(
        int32_t n_vocab,
        const std::vector<llama_ngram_bias> & pats,
        const std::vector<llama_token> & hist,
        bool * sorted_out = nullptr,
        int * selected_out = nullptr,
        bool sorted_in = false) {
    llama_sampler * smpl = llama_sampler_init_ngram_bias(n_vocab, (int32_t) pats.size(), pats.empty() ? nullptr : pats.data());
    GGML_ASSERT(smpl != nullptr);
    for (auto t : hist) {
        llama_sampler_accept(smpl, t);
    }
    std::vector<llama_token_data> cur;
    for (llama_token i = 0; i < n_vocab; ++i) {
        cur.push_back({i, 0.0f, 0.0f});
    }
    llama_token_data_array cur_p = {cur.data(), cur.size(), -1, sorted_in};
    llama_sampler_apply(smpl, &cur_p);
    std::vector<float> out(cur_p.size);
    for (size_t i = 0; i < cur_p.size; ++i) {
        out[i] = cur_p.data[i].logit;
    }
    if (sorted_out != nullptr) {
        *sorted_out = cur_p.sorted ? 1 : 0;
    }
    if (selected_out != nullptr) {
        *selected_out = (int) cur_p.selected;
    }
    llama_sampler_free(smpl);
    return out;
}

static void test_ngram_bias() {
    const int32_t V = 10;
    // basic: ([1],2,-3.0) fires after 1
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -3.0f}};
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == -3.0f);
        for (int i = 0; i < V; ++i) if (i != 2) GGML_ASSERT(out[i] == 0.0f);
    }
    // reset / empty history: no-op
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -3.0f}};
        auto out = ngram_run(V, pats, {});
        for (int i = 0; i < V; ++i) GGML_ASSERT(out[i] == 0.0f);
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(smpl, 1);
        llama_sampler_reset(smpl);
        std::vector<llama_token_data> cur;
        for (llama_token i = 0; i < V; ++i) cur.push_back({i, 0.0f, 0.0f});
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
        llama_sampler_apply(smpl, &cur_p);
        for (size_t i = 0; i < cur_p.size; ++i) GGML_ASSERT(cur_p.data[i].logit == 0.0f);
        llama_sampler_free(smpl);
    }
    // no match on other prefix or short history
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -3.0f}};
        auto out = ngram_run(V, pats, {3});
        for (int i = 0; i < V; ++i) GGML_ASSERT(out[i] == 0.0f);
        llama_token t3[] = {1, 2, 3, 4};
        std::vector<llama_ngram_bias> p3 = {{t3, 4, -1.0f}};
        auto out2 = ngram_run(V, p3, {2, 3});
        for (int i = 0; i < V; ++i) GGML_ASSERT(out2[i] == 0.0f);
    }
    // only the suffix is ever biased, never the prefix itself
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -3.0f}};
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == -3.0f);
        GGML_ASSERT(out[1] == 0.0f);
    }
    // overlap additive + order independence (duplicate merge)
    {
        llama_token a[] = {1, 2};
        llama_token b[] = {1, 2};
        std::vector<llama_ngram_bias> p1 = {{a, 2, -1.0f}, {b, 2, -2.0f}};
        std::vector<llama_ngram_bias> p2 = {{b, 2, -2.0f}, {a, 2, -1.0f}};
        auto o1 = ngram_run(V, p1, {1});
        auto o2 = ngram_run(V, p2, {1});
        GGML_ASSERT(o1[2] == -3.0f);
        GGML_ASSERT(o2[2] == -3.0f);
    }
    // ban fires; -INFINITY absorbs a finite duplicate of the same pattern
    {
        llama_token a[] = {1, 2};
        llama_token c[] = {3, 2};
        std::vector<llama_ngram_bias> pats = {{a, 2, -INFINITY}, {c, 2, -1.0f}};
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == -INFINITY);
        auto out_none = ngram_run(V, pats, {3});
        GGML_ASSERT(out_none[2] == -1.0f);
        // same pattern twice (ban + finite) merges to -INFINITY
        std::vector<llama_ngram_bias> merged = {{a, 2, -INFINITY}, {a, 2, -1.0f}};
        auto out_m = ngram_run(V, merged, {1});
        GGML_ASSERT(out_m[2] == -INFINITY);
    }
    // longer prefix needs full match
    {
        llama_token t[] = {4, 5, 6, 7};
        std::vector<llama_ngram_bias> pats = {{t, 4, -2.0f}};
        auto miss = ngram_run(V, pats, {5, 6});
        GGML_ASSERT(miss[7] == 0.0f);
        auto hit = ngram_run(V, pats, {4, 5, 6});
        GGML_ASSERT(hit[7] == -2.0f);
    }
    // wildcard [1,-1,2]
    {
        llama_token t[] = {1, -1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 3, -2.0f}};
        for (llama_token x = 0; x < V; ++x) {
            auto out = ngram_run(V, pats, {1, x});
            GGML_ASSERT(out[2] == -2.0f);
        }
        auto short_h = ngram_run(V, pats, {1});
        GGML_ASSERT(short_h[2] == 0.0f);
        auto long_h = ngram_run(V, pats, {1, 3, 4});
        GGML_ASSERT(long_h[2] == 0.0f);
    }
    // suffix wildcard rejected
    {
        llama_token bad[] = {1, -1};
        llama_token good[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{bad, 2, -5.0f}, {good, 2, -1.0f}};
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == -1.0f);
    }
    // validation: length-0/1, OOR id, id < -1, all-wild prefix, >2 wild, NaN, +inf, |b|>100 skipped
    {
        llama_token l1[] = {1};
        llama_token oor[] = {1, 99};
        llama_token neg[] = {-2, 2};
        llama_token allw[] = {-1, 2};
        llama_token w3[] = {-1, -1, -1, 2};
        llama_token good[] = {1, 2};
        float nan_b = nanf("");
        float inf_b = INFINITY;
        std::vector<llama_ngram_bias> pats = {
            {l1, 1, -1.0f},
            {good, 0, -1.0f},
            {oor, 2, -1.0f},
            {neg, 2, -1.0f},
            {allw, 2, -1.0f},
            {w3, 4, -1.0f},
            {good, 2, nan_b},
            {good, 2, inf_b},
            {good, 2, 101.0f},
            {good, 2, -1.0f},
        };
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == -1.0f);
    }
    // isolated skips are observable as the empty-sampler marker
    {
        llama_token t_oor[] = {1, 99};
        llama_token t_len1[] = {1};
        llama_token t_allw[] = {-1, 2};
        llama_token t_allw2[] = {-1, -1, 2};
        const std::vector<llama_ngram_bias> cases[] = {
            {{t_oor, 2, -1.0f}},
            {{t_len1, 1, -1.0f}},
            {{t_allw, 2, -1.0f}},
            {{t_allw2, 3, -1.0f}},
        };
        for (const auto & p : cases) {
            llama_sampler * s = llama_sampler_init_ngram_bias(V, 1, p.data());
            GGML_ASSERT(std::string(llama_sampler_name(s)).find("?ngram-bias") != std::string::npos);
            llama_sampler_free(s);
        }
        // all-wildcard prefix with 2 wildcards also skips when history would satisfy it
        std::vector<llama_ngram_bias> p_aw = {{t_allw2, 3, -1.0f}};
        GGML_ASSERT(ngram_run(V, p_aw, {7, 8})[2] == 0.0f);
    }
    // max length 8 fires at full depth, length 9 rejected
    {
        llama_token t8[] = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<llama_ngram_bias> pats = {{t8, 8, -2.0f}};
        auto hit = ngram_run(V, pats, {0, 1, 2, 3, 4, 5, 6});
        GGML_ASSERT(hit[7] == -2.0f);
        auto miss = ngram_run(V, pats, {0, 1, 2, 3, 4, 5});
        GGML_ASSERT(miss[7] == 0.0f);
        // length 9 rejected even when the history would satisfy it
        // (a missing cap would fire here: n_max 9, ring 8, prefix of eight 5s)
        llama_token t9[] = {5, 5, 5, 5, 5, 5, 5, 5, 9};
        std::vector<llama_ngram_bias> p9 = {{t9, 9, -5.0f}};
        auto rej = ngram_run(V, p9, {5, 5, 5, 5, 5, 5, 5, 5});
        GGML_ASSERT(rej[9] == 0.0f);
    }
    // exactly 2 wildcards (max) accepted
    {
        llama_token t[] = {1, -1, -1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 4, -2.0f}};
        auto out = ngram_run(V, pats, {1, 7, 8});
        GGML_ASSERT(out[2] == -2.0f);
        auto miss = ngram_run(V, pats, {1, 7});
        GGML_ASSERT(miss[2] == 0.0f);
    }
    // |bias| == 100 accepted (only > 100 skipped)
    {
        llama_token a[] = {1, 2};
        llama_token b[] = {3, 4};
        std::vector<llama_ngram_bias> pats = {{a, 2, 100.0f}, {b, 2, -100.0f}};
        auto o1 = ngram_run(V, pats, {1});
        GGML_ASSERT(o1[2] == 100.0f);
        auto o2 = ngram_run(V, pats, {3});
        GGML_ASSERT(o2[4] == -100.0f);
    }
    // >2 wildcards rejected even when history would satisfy them
    {
        llama_token w3[] = {-1, -1, -1, 2};
        std::vector<llama_ngram_bias> pats = {{w3, 4, -1.0f}};
        auto out = ngram_run(V, pats, {7, 8, 1});
        GGML_ASSERT(out[2] == 0.0f);
    }
    // id 0 is a normal token
    {
        llama_token t[] = {0, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -2.0f}};
        auto hit = ngram_run(V, pats, {0});
        GGML_ASSERT(hit[2] == -2.0f);
        auto miss = ngram_run(V, pats, {1});
        GGML_ASSERT(miss[2] == 0.0f);
    }
    // leading wildcard with concrete backup allowed (L>=3)
    {
        llama_token t[] = {-1, 1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 3, -2.0f}};
        auto out = ngram_run(V, pats, {9, 1});
        GGML_ASSERT(out[2] == -2.0f);
    }
    // integer bias accepted, zero bias no-op
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -2}};
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == -2.0f);
        std::vector<llama_ngram_bias> pz = {{t, 2, 0.0f}};
        auto oz = ngram_run(V, pz, {1});
        GGML_ASSERT(oz[2] == 0.0f);
    }
    // clone shares trie, copies hist; copy_state restores hist
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -3.0f}};
        llama_sampler * a = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(a, 1);
        llama_sampler * b = llama_sampler_clone(a);
        llama_sampler_accept(a, 5);
        std::vector<llama_token_data> cur;
        for (llama_token i = 0; i < V; ++i) cur.push_back({i, 0.0f, 0.0f});
        llama_token_data_array pa = {cur.data(), cur.size(), -1, false};
        llama_sampler_apply(b, &pa);
        GGML_ASSERT(pa.data[2].logit == -3.0f);
        llama_sampler_copy(a, b);
        std::vector<llama_token_data> cur2;
        for (llama_token i = 0; i < V; ++i) cur2.push_back({i, 0.0f, 0.0f});
        llama_token_data_array pb = {cur2.data(), cur2.size(), -1, false};
        llama_sampler_apply(b, &pb);
        GGML_ASSERT(pb.data[2].logit == 0.0f);
        llama_sampler_free(a);
        llama_sampler_free(b);
    }
    // empty init robustness
    {
        llama_sampler * s1 = llama_sampler_init_ngram_bias(V, 0, nullptr);
        GGML_ASSERT(s1 != nullptr);
        llama_sampler_free(s1);
        llama_sampler * s2 = llama_sampler_init_ngram_bias(0, 0, nullptr);
        GGML_ASSERT(s2 != nullptr);
        llama_sampler_free(s2);
        llama_ngram_bias bad = {nullptr, 2, -1.0f};
        llama_sampler * s3 = llama_sampler_init_ngram_bias(V, 1, &bad);
        GGML_ASSERT(s3 != nullptr);
        GGML_ASSERT(std::string(llama_sampler_name(s3)).find("?ngram-bias") != std::string::npos);
        llama_sampler_free(s3);
        llama_sampler * s3b = llama_sampler_init_ngram_bias(-3, 1, nullptr);
        GGML_ASSERT(s3b != nullptr);
        llama_sampler_free(s3b);
        llama_sampler * s4 = llama_sampler_init_ngram_bias(V, -5, nullptr);
        GGML_ASSERT(s4 != nullptr);
        llama_sampler_free(s4);
        llama_sampler * s5 = llama_sampler_init_ngram_bias(V, 5, nullptr);
        GGML_ASSERT(s5 != nullptr);
        llama_sampler_free(s5);
    }
    // diff-len overlaps fire at every depth
    {
        llama_token s[] = {1, 2};
        llama_token l[] = {3, 1, 2};
        std::vector<llama_ngram_bias> pats = {{s, 2, -1.0f}, {l, 3, -2.0f}};
        auto both = ngram_run(V, pats, {3, 1});
        GGML_ASSERT(both[2] == -3.0f);
        auto short_only = ngram_run(V, pats, {9, 1});
        GGML_ASSERT(short_only[2] == -1.0f);
    }
    // history overflow: n_max=3 cap 2, 20 accepts ending in prefix
    {
        llama_token t[] = {7, 8, 9};
        std::vector<llama_ngram_bias> pats = {{t, 3, -2.0f}};
        std::vector<llama_token> hist;
        for (int i = 0; i < 18; ++i) hist.push_back(0);
        hist.push_back(7);
        hist.push_back(8);
        auto out = ngram_run(V, pats, hist);
        GGML_ASSERT(out[9] == -2.0f);
        std::vector<llama_token> stale;
        for (int i = 0; i < 10; ++i) stale.push_back(7);
        stale.push_back(8);
        for (int i = 0; i < 10; ++i) stale.push_back(0);
        auto out2 = ngram_run(V, pats, stale);
        GGML_ASSERT(out2[9] == 0.0f);
    }
    // suffix absent from candidates: skip silently
    {
        llama_token t[] = {1, 9};
        std::vector<llama_ngram_bias> pats = {{t, 2, -2.0f}};
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(smpl, 1);
        std::vector<llama_token_data> cur;
        for (llama_token i = 0; i < 5; ++i) cur.push_back({i, 0.0f, 0.0f});
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
        llama_sampler_apply(smpl, &cur_p);
        GGML_ASSERT(cur_p.size == 5);
        for (size_t i = 0; i < cur_p.size; ++i) GGML_ASSERT(cur_p.data[i].logit == 0.0f);
        llama_sampler_free(smpl);
    }
    // slow path: suffix found by id scan when index and id differ
    {
        llama_token t[] = {1, 9};
        std::vector<llama_ngram_bias> pats = {{t, 2, -2.0f}};
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(smpl, 1);
        std::vector<llama_token_data> cur;
        for (llama_token i = V - 1; i >= 0; --i) cur.push_back({i, 0.0f, 0.0f});
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
        llama_sampler_apply(smpl, &cur_p);
        for (size_t i = 0; i < cur_p.size; ++i) {
            if (cur_p.data[i].id == 9) {
                GGML_ASSERT(cur_p.data[i].logit == -2.0f);
            } else {
                GGML_ASSERT(cur_p.data[i].logit == 0.0f);
            }
        }
        llama_sampler_free(smpl);
    }
    // C-layer truncation: only the first 2048 patterns are used
    {
        std::vector<std::vector<llama_token>> store(2049, std::vector<llama_token>{7, 8});
        store[2048] = {1, 2};
        std::vector<llama_ngram_bias> pats;
        pats.reserve(2049);
        for (int i = 0; i < 2049; ++i) pats.push_back({store[i].data(), 2, -1.0f});
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == 0.0f);
        auto out2 = ngram_run(V, pats, {7});
        GGML_ASSERT(out2[8] == -2048.0f);
    }
    // determinism: identical apply twice gives identical logits
    {
        llama_token t[] = {1, -1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 3, -2.0f}};
        auto o1 = ngram_run(V, pats, {1, 5});
        auto o2 = ngram_run(V, pats, {1, 5});
        GGML_ASSERT(o1 == o2);
        // same sampler, fresh candidates: scratch reuse does not corrupt
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(smpl, 1);
        llama_sampler_accept(smpl, 5);
        std::vector<llama_token_data> c1;
        for (llama_token i = 0; i < V; ++i) c1.push_back({i, 0.0f, 0.0f});
        llama_token_data_array p1 = {c1.data(), c1.size(), -1, false};
        llama_sampler_apply(smpl, &p1);
        std::vector<llama_token_data> c2;
        for (llama_token i = 0; i < V; ++i) c2.push_back({i, 0.0f, 0.0f});
        llama_token_data_array p2 = {c2.data(), c2.size(), -1, false};
        llama_sampler_apply(smpl, &p2);
        for (size_t i = 0; i < p1.size; ++i) GGML_ASSERT(p1.data[i].logit == p2.data[i].logit);
        llama_sampler_free(smpl);
    }
    // finite bias never resurrects a pre-banned logit
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -1.0f}};
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(smpl, 1);
        std::vector<llama_token_data> cur;
        for (llama_token i = 0; i < V; ++i) cur.push_back({i, i == 2 ? -INFINITY : 0.0f, 0.0f});
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
        llama_sampler_apply(smpl, &cur_p);
        GGML_ASSERT(cur_p.data[2].logit == -INFINITY);
        llama_sampler_free(smpl);
    }
    // ban guard: bans never take the last finite logit, so no all -INFINITY deadlock
    {
        // ban every continuation after 1: exactly one survivor stays finite
        std::vector<std::vector<llama_token>> store;
        std::vector<llama_ngram_bias> pats;
        for (llama_token i = 0; i < V; ++i) {
            store.push_back({1, i});
            pats.push_back({store.back().data(), 2, -INFINITY});
        }
        bool sorted = true;
        auto out = ngram_run(V, pats, {1}, &sorted, nullptr, true);
        int n_finite = 0;
        for (int i = 0; i < V; ++i) {
            if (out[i] != -INFINITY) {
                n_finite++;
                GGML_ASSERT(out[i] == 0.0f);
            }
        }
        GGML_ASSERT(n_finite == 1);
        GGML_ASSERT(sorted == false);
        // deterministic: same survivor across runs (merged map order bans 0..V-2, spares V-1)
        GGML_ASSERT(out[V - 1] == 0.0f);
        auto out2 = ngram_run(V, pats, {1});
        GGML_ASSERT(out == out2);
        // same survivor via the id-scan path (reversed candidates)
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, (int32_t) pats.size(), pats.data());
        llama_sampler_accept(smpl, 1);
        std::vector<llama_token_data> cur;
        for (llama_token i = V - 1; i >= 0; --i) cur.push_back({i, 0.0f, 0.0f});
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
        llama_sampler_apply(smpl, &cur_p);
        int surv = -1;
        for (size_t i = 0; i < cur_p.size; ++i) {
            if (cur_p.data[i].logit != -INFINITY) {
                GGML_ASSERT(surv == -1);
                surv = cur_p.data[i].id;
            }
        }
        GGML_ASSERT(surv == V - 1);
        llama_sampler_free(smpl);
    }
    // partial bans still land fully when survivors remain
    {
        llama_token a[] = {1, 2};
        llama_token b[] = {1, 3};
        std::vector<llama_ngram_bias> pats = {{a, 2, -INFINITY}, {b, 2, -INFINITY}};
        auto out = ngram_run(V, pats, {1});
        GGML_ASSERT(out[2] == -INFINITY);
        GGML_ASSERT(out[3] == -INFINITY);
        for (int i = 0; i < V; ++i) {
            if (i != 2 && i != 3) GGML_ASSERT(out[i] == 0.0f);
        }
    }
    // redundant-only bans change nothing, not even the sorted flag
    {
        llama_token a[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{a, 2, -INFINITY}};
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(smpl, 1);
        std::vector<llama_token_data> cur;
        for (llama_token i = 0; i < V; ++i) cur.push_back({i, i == 2 ? -INFINITY : 0.0f, 0.0f});
        llama_token_data_array cur_p = {cur.data(), cur.size(), -1, true};
        llama_sampler_apply(smpl, &cur_p);
        GGML_ASSERT(cur_p.data[2].logit == -INFINITY);
        GGML_ASSERT(cur_p.sorted == true);
        llama_sampler_free(smpl);
    }
    // name and purity: no selected, no RNG, sorted=false on hit only
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -1.0f}};
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        GGML_ASSERT(std::string(llama_sampler_name(smpl)).find("ngram") != std::string::npos);
        llama_sampler_free(smpl);
        bool sorted = true;
        int sel = 5;
        ngram_run(V, pats, {1}, &sorted, &sel, true);
        GGML_ASSERT(sorted == false);
        GGML_ASSERT(sel == -1);
        // miss leaves a pre-sorted array untouched
        bool sorted_miss = true;
        ngram_run(V, pats, {3}, &sorted_miss, nullptr, true);
        GGML_ASSERT(sorted_miss == true);
        // miss never touches a preset selected index
        {
            llama_sampler * m = llama_sampler_init_ngram_bias(V, 1, pats.data());
            llama_sampler_accept(m, 3);
            std::vector<llama_token_data> cur;
            for (llama_token i = 0; i < V; ++i) cur.push_back({i, 0.0f, 0.0f});
            llama_token_data_array mp = {cur.data(), cur.size(), 7, true};
            llama_sampler_apply(m, &mp);
            GGML_ASSERT(mp.selected == 7);
            GGML_ASSERT(mp.sorted == true);
            llama_sampler_free(m);
        }
        llama_sampler * e = llama_sampler_init_ngram_bias(V, 0, nullptr);
        GGML_ASSERT(std::string(llama_sampler_name(e)).find("?ngram-bias") != std::string::npos);
        llama_sampler_free(e);
    }
    // defensive accept: invalid (-1) and OOR ids never reach history,
    // else a stored one would match a wildcard edge
    {
        llama_token t[] = {1, -1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 3, -1.0f}};
        const llama_token bad_ids[] = {-1, 99};
        for (llama_token bad : bad_ids) {
            llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
            llama_sampler_accept(smpl, 1);
            llama_sampler_accept(smpl, bad);
            std::vector<llama_token_data> cur;
            for (llama_token i = 0; i < V; ++i) cur.push_back({i, 0.0f, 0.0f});
            llama_token_data_array cur_p = {cur.data(), cur.size(), -1, false};
            llama_sampler_apply(smpl, &cur_p);
            for (size_t i = 0; i < cur_p.size; ++i) GGML_ASSERT(cur_p.data[i].logit == 0.0f);
            llama_sampler_free(smpl);
        }
    }
    // empty candidates early return
    {
        llama_token t[] = {1, 2};
        std::vector<llama_ngram_bias> pats = {{t, 2, -1.0f}};
        llama_sampler * smpl = llama_sampler_init_ngram_bias(V, 1, pats.data());
        llama_sampler_accept(smpl, 1);
        llama_token_data_array cur_p = {nullptr, 0, -1, false};
        llama_sampler_apply(smpl, &cur_p);
        llama_sampler_free(smpl);
    }
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

int main(void) {
    ggml_time_init();

    test_dist_singleton_rng();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    test_ngram_bias();

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    printf("OK\n");

    test_perf();

    return 0;
}
