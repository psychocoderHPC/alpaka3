#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#define USE_ALPAKA 1

struct tableau
{
    static constexpr size_t nEntries = 6;

    static constexpr auto entries_low = alpaka::Vec{25 / 216.0, 0.0, 1408 / 2565.0, 2197 / 4104.0, -0.2, 0.0};

    static constexpr auto entries_high
        = alpaka::Vec{16 / 135.0, 0.0, 6656 / 12825.0, 28561 / 56430.0, -9 / 50.0, 2 / 55.0};

    static constexpr auto entries = alpaka::Vec{
        alpaka::Vec<double, 6>{0.25, 0.25},
        alpaka::Vec<double, 6>{3 / 8.0, 3 / 32.0, 9 / 32.0},
        alpaka::Vec<double, 6>{12 / 13.0, 1932 / 2197.0, -7200 / 2197.0, 7296 / 2197.0},
        alpaka::Vec<double, 6>{1.0, 439 / 216.0, -8.0, 3680 / 513.0, -845 / 4104.0},
        alpaka::Vec<double, 6>{0.5, -8 / 27.0, 2.0, -3544 / 2565.0, 1859 / 4104.0, -11 / 40.0}};
};

void print(std::vector<double>& vec)
{
    for(auto& val : vec)
    {
        printf("%e ", val);
    }
    std::cout << "\n";
}

void print(std::vector<std::vector<double>>& vec)
{
    for(auto& val : vec)
    {
        print(val);
    }
}

template<typename T_Exec, typename T_Queue, typename T_Buffer,typename T_Buffer_dev>
struct Monstrosity
{
    T_Exec exec;
    T_Queue queue;
    double abs_tol, rel_tol, dt_min, dt_max;

    std::vector<double> m_yt_eval, m_error_estimate;
    T_Buffer m_kt_values; // col major!
    T_Buffer_dev m_kt_values_dev;

    tableau tab;

    Monstrosity(
        T_Exec in_exec,
        T_Queue in_queue,
        double in_abs_tol,
        double in_rel_tol,
        double in_dt_min,
        double in_dt_max,
        std::vector<double> yt_eval,
        std::vector<double> error_estimate,
        T_Buffer& kt_values,T_Buffer_dev& kt_values_dev)
        : exec{in_exec}
        , queue{in_queue}
        , abs_tol{in_abs_tol}
        , rel_tol{in_rel_tol}
        , dt_min{in_dt_min}
        , dt_max{in_dt_max}
        , m_yt_eval{std::move(yt_eval)}
        , m_error_estimate{std::move(error_estimate)}
        , m_kt_values{kt_values}, m_kt_values_dev{kt_values_dev}
    {
        alpaka::onHost::memset(queue, m_kt_values, 0);
        alpaka::onHost::memcpy(queue,m_kt_values_dev,m_kt_values);
        alpaka::onHost::wait(queue);
    }

    // using DerivFunction = void (*)(std::vector<double> const& y, double t, std::vector<double>& dydt);

    bool step(
        auto const f,
        std::vector<double> const& yt,
        double& t,
        double& dt,
        std::vector<double>& ytp1,
        auto amplitude_lincomb,
        auto t_offset,
        auto t_scale)
    {
        assert(0 <= dt_min);
        assert(dt_min <= dt_max);

        if(dt < dt_min || dt > dt_max)
        {
            // mio::log_warning("IntegratorCore: Restricting given step size dt = {} to [{}, {}].", dt, dt_min,
            // dt_max);
        }

        dt = std::min(dt, dt_max);

        double t_eval; // shifted time for evaluating yt
        double dt_new; // updated dt

        bool converged = false; // carry for convergence criterion
        bool dt_is_invalid = false;

        // if (m_yt_eval.size() != yt.size()) {
        //     m_yt_eval.resize(yt.size());
        //     m_kt_values.resize(yt.size(), tab.entries_low.size());
        // }

        for(size_t j = 1; j < yt.size(); j++)
        {
            m_yt_eval[j] = yt[j];
        }

        while(!converged && !dt_is_invalid)
        {
            if(dt < dt_min)
            {
                dt_is_invalid = true;
                dt = dt_min;
            }
            // std::cout << "---- step t:" << t << " dt:" << dt << "\n";
            // std::cin.ignore();
            // compute first column of kt, i.e. kt_0 for each y in yt_eval
            constexpr size_t frame_extent = 256;
            queue.enqueue(
                exec,
                alpaka::onHost::FrameSpec{alpaka::divExZero(m_yt_eval.size(), frame_extent), frame_extent},
                alpaka::KernelBundle{
                    f,
                    m_yt_eval.size(),
                    t,
                    m_kt_values_dev.getMdSpan(),
                    amplitude_lincomb.getMdSpan(),
                    t_offset.getMdSpan(),
                    t_scale.getMdSpan(),
                    0});
            alpaka::onHost::memcpy(queue,m_kt_values,m_kt_values_dev);
            alpaka::onHost::wait(queue);

            for(size_t i = 1; i < m_kt_values.getExtents().y(); i++)
            {
                // we first compute k_n1 for each y_j, then k_n2 for each y_j, etc.
                t_eval = t;
                t_eval += tab.entries[i - 1][0]
                          * dt; // t_eval = t + c_i * h // note: line zero of Butcher tableau not stored in array
                // use ytp1 as temporary storage for evaluating m_kt_values[i]
                ytp1 = m_yt_eval;
                for(size_t k = 1; k < ALPAKA_TYPEOF(tab.entries[i - 1])::dim(); k++)
                {
                    for(size_t j = 1; j < yt.size(); j++)
                    {
                        ytp1[j] += (dt * tab.entries[i - 1][k]) * m_kt_values[alpaka::Vec{k - 1, j}];
                    }
                }
                // get the derivatives, i.e., compute kt_i for all y in ytp1: kt_i = f(t_eval, ytp1low)
                queue.enqueue(
                    exec,
                    alpaka::onHost::FrameSpec{alpaka::divExZero(ytp1.size(), frame_extent), frame_extent},
                    alpaka::KernelBundle{
                        f,
                        ytp1.size(),
                        t_eval,
                        m_kt_values_dev.getMdSpan(),
                        amplitude_lincomb.getMdSpan(),
                        t_offset.getMdSpan(),
                        t_scale.getMdSpan(),
                        i});
                alpaka::onHost::memcpy(queue,m_kt_values,m_kt_values_dev);
                alpaka::onHost::wait(queue);
            }

            // for (int i = 0; i < 6; i++) {
            //     std::cout << "eval (" << i << "): ";
            //     print(m_kt_values[i]);
            // }
            // calculate low order estimate

            for(size_t i = 0; i < yt.size(); i++)
            {
                ytp1[i] = m_yt_eval[i];
                for(size_t j = 0; j < m_kt_values.getExtents().y(); j++)
                {
                    ytp1[i] += (dt * (m_kt_values[alpaka::Vec{j, i}] * tab.entries_low[j]));
                }
            }
            // std::cout << "low: "; print(ytp1);
            // truncation error estimate: yt_low - yt_high = O(h^(p+1)) where p = order of convergence
            // #pragma acc parallel loop
            for(size_t i = 0; i < yt.size(); i++)
            {
                m_error_estimate[i] = 0;
                for(size_t j = 0; j < m_kt_values.getExtents().y(); j++)
                {
                    m_error_estimate[i]
                        += dt * m_kt_values[alpaka::Vec{j, i}] * (tab.entries_high[j] - tab.entries_low[j]);
                }
                m_error_estimate[i] = std::abs(m_error_estimate[i]);
            }

            // std::cout << "tab diff: ";
            // for (int j = 0; j < 6; j++) {
            //     std::cout << tab.entries_high[j] - tab.entries_low[j] << " ";
            // }
            // std::cout << "\n";

            // std::cout << "kt * tab: ";
            // for (int i = 0; i < yt.size(); i++) {
            //     double x = 0;
            //     for (int j = 0; j < 6; j++) {
            //         x += m_kt_values[j][i] * (tab.entries_high[j] - tab.entries_low[j]);
            //     }
            //     std::cout << x << " ";
            // }
            // std::cout << "\n";

            // std::cout << "err: "; print(m_error_estimate);
            // calculate mixed tolerance

            double min_coeff = std::numeric_limits<double>::infinity();
            // std::cout << "con: ";
            // #pragma acc parallel loop
            for(size_t i = 0; i < yt.size(); i++)
            {
                double tmp = (abs_tol + std::abs(ytp1[i]) * rel_tol) / m_error_estimate[i];
                // std::cout << tmp << " ";
                min_coeff = std::min(min_coeff, tmp);
            }
            converged = (min_coeff >= 1);
            // std::cout << "\nmin con: " << min_coeff << "\n";
            // converged = (min_coeff <= 1); // convergence criterion


            if(converged || dt_is_invalid)
            {
                // if sufficiently exact, return ytp1, which currently contains the lower order approximation
                // (higher order is not always higher accuracy)
                t += dt; // this is the t where ytp1 belongs to
            }
            // else: repeat the calculation above (with updated dt)

            // compute new value for dt
            // converged implies eps/error_estimate >= 1, so dt will be increased for the next step
            // hence !converged implies 0 < eps/error_estimate < 1, strictly decreasing dt
            dt_new = dt * std::pow(min_coeff, (1. / (ALPAKA_TYPEOF(tab.entries_low)::dim() - 1)));
            // safety factor for more conservative step increases,
            // and to avoid dt_new -> dt for step decreases when |error_estimate - eps| -> 0
            dt_new *= 0.9;
            // std::cout << "dt: " << dt << " dt_new: " << dt_new << "\n";
            // check if updated dt stays within desired bounds and update dt for next step
            dt = std::min(dt_new, dt_max);
        }
        dt = std::max(dt, dt_min);
        // return 'converged' in favor of '!dt_is_invalid', as these values only differ if step sizing failed,
        // but the step with size dt_min was accepted.
        return converged;
    }
};
