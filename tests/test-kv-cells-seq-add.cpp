/*
 * test-kv-cells-seq-add.cpp — Unit test for llama_kv_cells position
 * shifting (pos_add / pos_div), verifying correctness for IMROPE
 * models where a single scalar pos is broadcast to 4 ROPE dimensions.
 *
 * Tests the cells layer directly — no model load required.
 */
#include "llama-kv-cells.h"

#include <cassert>
#include <cstdio>
#include <vector>

// Simulate the IMROPE 4D broadcast from llama-graph.cpp:103-112
static std::vector<llama_pos> imrope_broadcast(llama_pos pos) {
    return {pos, pos, pos, 0};
}

static int n_fail = 0;

static void check(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        n_fail++;
    }
}

static void test_pos_add_basic() {
    printf("--- test_pos_add_basic ---\n");
    llama_kv_cells cells;
    cells.resize(8);

    // Populate cells 0-4 with positions 10-14, all on seq 0
    for (int i = 0; i < 5; i++) {
        cells.pos_set(i, 10 + i);
        cells.seq_add(i, 0);
    }

    check(cells.pos_get(0) == 10, "initial pos[0] == 10");
    check(cells.pos_get(4) == 14, "initial pos[4] == 14");

    // Shift cells 2-4 by -2 (simulates spec decode draft reuse)
    for (int i = 2; i < 5; i++) {
        cells.pos_add(i, -2);
    }

    check(cells.pos_get(0) == 10, "after shift pos[0] unchanged");
    check(cells.pos_get(1) == 11, "after shift pos[1] unchanged");
    check(cells.pos_get(2) == 10, "after shift pos[2] == 10 (was 12, -2)");
    check(cells.pos_get(3) == 11, "after shift pos[3] == 11 (was 13, -2)");
    check(cells.pos_get(4) == 12, "after shift pos[4] == 12 (was 14, -2)");

    // Verify IMROPE broadcast produces correct 4D positions after shift
    auto dims = imrope_broadcast(cells.pos_get(2));
    check(dims[0] == 10, "IMROPE dim0 == 10");
    check(dims[1] == 10, "IMROPE dim1 == 10");
    check(dims[2] == 10, "IMROPE dim2 == 10");
    check(dims[3] == 0,  "IMROPE dim3 == 0");

    printf("  pos_add_basic: %s\n", n_fail == 0 ? "PASS" : "FAIL");
}

static void test_pos_add_negative_clears() {
    printf("--- test_pos_add_negative_clears ---\n");
    int prev_fail = n_fail;
    llama_kv_cells cells;
    cells.resize(4);

    cells.pos_set(0, 5);
    cells.seq_add(0, 0);

    // Shift by -10 makes pos negative → cell should be cleared
    bool cleared = cells.pos_add(0, -10);
    check(cleared, "pos_add returns true when pos goes negative");
    check(cells.is_empty(0), "cell cleared when pos < 0");

    printf("  pos_add_negative_clears: %s\n", n_fail == prev_fail ? "PASS" : "FAIL");
}

static void test_pos_div() {
    printf("--- test_pos_div ---\n");
    int prev_fail = n_fail;
    llama_kv_cells cells;
    cells.resize(4);

    cells.pos_set(0, 100);
    cells.seq_add(0, 0);

    cells.pos_div(0, 2);
    check(cells.pos_get(0) == 50, "pos_div by 2: 100 → 50");

    // IMROPE broadcast after div
    auto dims = imrope_broadcast(cells.pos_get(0));
    check(dims[0] == 50, "IMROPE dim0 == 50 after div");
    check(dims[3] == 0,  "IMROPE dim3 == 0 after div");

    printf("  pos_div: %s\n", n_fail == prev_fail ? "PASS" : "FAIL");
}

static void test_shift_tracking() {
    printf("--- test_shift_tracking ---\n");
    int prev_fail = n_fail;
    llama_kv_cells cells;
    cells.resize(4);

    cells.pos_set(0, 20);
    cells.seq_add(0, 0);

    check(!cells.get_has_shift(), "no shift initially");

    cells.pos_add(0, 5);
    check(cells.get_has_shift(), "has_shift after pos_add");
    check(cells.pos_get(0) == 25, "pos updated to 25");

    printf("  shift_tracking: %s\n", n_fail == prev_fail ? "PASS" : "FAIL");
}

static void test_seq_isolation() {
    printf("--- test_seq_isolation ---\n");
    int prev_fail = n_fail;
    llama_kv_cells cells;
    cells.resize(4);

    // Cell 0: seq 0, pos 10
    // Cell 1: seq 1, pos 20
    cells.pos_set(0, 10); cells.seq_add(0, 0);
    cells.pos_set(1, 20); cells.seq_add(1, 1);

    // Shift only cells matching seq 0 (manually, as seq_add would)
    for (uint32_t i = 0; i < cells.size(); i++) {
        if (!cells.is_empty(i) && cells.seq_has(i, 0)) {
            cells.pos_add(i, -3);
        }
    }

    check(cells.pos_get(0) == 7,  "seq 0 shifted: 10 → 7");
    check(cells.pos_get(1) == 20, "seq 1 unchanged: 20");

    printf("  seq_isolation: %s\n", n_fail == prev_fail ? "PASS" : "FAIL");
}

int main() {
    printf("=== test-kv-cells-seq-add ===\n\n");

    test_pos_add_basic();
    test_pos_add_negative_clears();
    test_pos_div();
    test_shift_tracking();
    test_seq_isolation();

    printf("\n%d failures.\n", n_fail);
    if (n_fail > 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS: all KV cell position shift tests passed.\n");
    return 0;
}
