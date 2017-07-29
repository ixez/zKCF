#pragma once
namespace zkcf {
    class IKernel {
    public:
        typedef enum
        {
            GAUSSIAN    = 1,
            POLYNOMIAL  = 2,
            LINEAR      = 3
        } Type;
    };
}