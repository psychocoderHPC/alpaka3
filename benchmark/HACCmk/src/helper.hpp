
#include <unistd.h>

#include <cstdint>
#include <iostream>

namespace hacc
{
    /* Vector length, must be divisible by 4  15000 */
    constexpr size_t defaultProblemSize = 15000;

    void help(char* argv[])
    {
        std::cerr << argv[0] << " [-n  numElements] [-h]" << std::endl;
    }

    int parseCmd(int argc, char* argv[], size_t& numElements, size_t& numberOfRuns)
    {
        numElements = defaultProblemSize;
        numberOfRuns = 1;

        int opt;
        while((opt = getopt(argc, argv, "hn:r:")) != -1)
        {
            switch(opt)
            {
            case 'n':
                try
                {
                    numElements = std::stoul(optarg, nullptr, 0);
                    // round to a multiple of 4
                    numElements = (numElements + 3u) / 4u * 4u;
                }
                catch(std::invalid_argument const& e)
                {
                    std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                    return EXIT_FAILURE;
                }
                catch(std::out_of_range const& e)
                {
                    std::cerr << "Error: value '" << optarg << "' out of range for size_t.\n";
                    return EXIT_FAILURE;
                }
                break;
            case 'r':
                try
                {
                    numberOfRuns = std::stoul(optarg, nullptr, 0);
                }
                catch(std::invalid_argument const& e)
                {
                    std::cerr << "Error: invalid number of runs '" << optarg << "'.\n";
                    return EXIT_FAILURE;
                }
                catch(std::out_of_range const& e)
                {
                    std::cerr << "Error: number of runs '" << optarg << "' out of range for size_t.\n";
                    return EXIT_FAILURE;
                }
                if(numberOfRuns == 0)
                {
                    std::cerr << "Error: number of runs must be greater than zero.\n";
                    return EXIT_FAILURE;
                }
                break;
            case 'h':
                help(argv);
                return EXIT_FAILURE;
            default:
                help(argv);
                return EXIT_FAILURE;
            }
        }

        return EXIT_SUCCESS;
    }

    template<typename T>
    inline auto* allocAlligned(size_t numElements)
    {
        constexpr size_t alignmentBytes = 256;
        return reinterpret_cast<T*>(::operator new(numElements * sizeof(T), std::align_val_t{alignmentBytes}));
    }

} // namespace hacc
