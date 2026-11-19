#include <acl/acl.h>
#include <cstdio>
#define CK(expr) do { if (auto code = (expr); code != ACL_SUCCESS) { \
    std::printf("%s failed: %d\n", #expr, int(code)); return 1; } } while (0)
bool load(const char* name, void* dest, size_t size) {
    auto* f = std::fopen(name, "rb");
    if (!f) return false;
    auto* data = new unsigned char[size];
    bool ok = std::fread(data, 1, size, f) == size;
    std::fclose(f);
    if (ok) ok = aclrtMemcpy(dest, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    delete[] data;
    return ok;
}
int main() {
    CK(aclInit(nullptr));
    CK(aclrtSetDevice(0));
    aclrtStream stream;
    CK(aclrtCreateStream(&stream));
    @ALLOCS@
    for (int launch = 0; launch < @REPEATS@; ++launch) {
        if (!load("@OUTPUT@.bin", d@OUTPUT@, size@OUTPUT@)) return 4;
        @KERNEL@<<<1, nullptr, stream>>>(@ARGS@);
        CK(aclrtSynchronizeStream(stream));
        auto* actual = new unsigned char[size@OUTPUT@];
        CK(aclrtMemcpy(actual, size@OUTPUT@, d@OUTPUT@, size@OUTPUT@, ACL_MEMCPY_DEVICE_TO_HOST));
        char name[64];
        std::snprintf(name, sizeof(name), "actual_%d.bin", launch + 1);
        auto* f = std::fopen(name, "wb");
        if (!f || std::fwrite(actual, 1, size@OUTPUT@, f) != size@OUTPUT@) return 5;
        std::fclose(f);
        delete[] actual;
    }
    @FREE@
    CK(aclrtDestroyStream(stream));
    CK(aclrtResetDevice(0));
    CK(aclFinalize());
    return 0;
}
