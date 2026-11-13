#include <acl/acl.h>
#include <cstdint>
#include <cstdio>

#define ACL_CHECK(expr) do { if (auto code = (expr); code != ACL_SUCCESS) { \
    std::printf("%s failed: %d\n", #expr, int(code)); return 1; } } while (0)

int main() {
    constexpr int N = 32, K = VERIFY_K;
    uint16_t x[K], w[N * K];
    float actual[N], expected[N] = {};
    constexpr uint16_t bits[] = {0x0000, 0x3c00, 0xbc00, 0x3800, 0xb400};
    constexpr float vals[] = {0, 1, -1, 0.5f, -0.25f};
    for (int k = 0; k < K; ++k) {
        const int xi = (k * 3 + k / 7 + k / 256) % 5;
        x[k] = bits[xi];
        for (int n = 0; n < N; ++n) {
            const int wi = (n * 7 + k * 2 + k / (n + 3)) % 5;
            w[n * K + k] = bits[wi];
            expected[n] += vals[xi] * vals[wi];
        }
    }
    for (float &value : actual) value = -12345;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    void *dx, *dw, *dy;
    ACL_CHECK(aclrtMalloc(&dx, sizeof(x), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dw, sizeof(w), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dy, sizeof(actual), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(dx, sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dw, sizeof(w), w, sizeof(w), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(dy, sizeof(actual), actual, sizeof(actual), ACL_MEMCPY_HOST_TO_DEVICE));
    main_kernel<<<1, nullptr, stream>>>(VERIFY_ARGS);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(actual, sizeof(actual), dy, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST));
    float max_error = 0;
    for (int n = 0; n < N; ++n) {
        float error = actual[n] - expected[n];
        if (error < 0) error = -error;
        if (!(error == 0)) {
            std::printf("FAIL n=%d got=%g expected=%g\n", n, actual[n], expected[n]);
            return 2;
        }
        if (error > max_error) max_error = error;
    }
    ACL_CHECK(aclrtFree(dx));
    ACL_CHECK(aclrtFree(dw));
    ACL_CHECK(aclrtFree(dy));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    std::printf("GEMV NUMERICAL PASS: N=%d K=%d max_error=%g\n", N, K, max_error);
    return 0;
}
