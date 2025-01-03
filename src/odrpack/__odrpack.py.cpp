#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

#include "odrpack.h"

namespace py = pybind11;
using namespace pybind11::literals;

/*
The following class is necessary to ensure that the static variables used to store the callback
functions are automatically reset upon normal or abnormal exit from the `odr_wrapper` function.
From: https://github.com/libprima/prima/blob/main/python/_prima.cpp
*/
class SelfCleaningPyObject {
    py::object &obj;

   public:
    SelfCleaningPyObject(py::object &obj) : obj(obj) {}
    ~SelfCleaningPyObject() { obj = py::none(); }
};

/*
Wrapper for the C-interface of the Fortran ODR routine. This wrapper is
intentionally very thin, with all argument checks and array dimension
calculations delegated to the companion Python caller, which serves as the entry
point for all function calls.

Some arguments have a default value of `nullptr` — this is by design, as the
Fortran code automatically interprets `nullptr` as an absent optional argument.
This approach avoids the redundant definition of default values in multiple
places.
*/
int odr_wrapper(int n, int m, int npar, int nq,
                int ldwe, int ld2we, int ldwd,
                int ld2wd, int ldifx, int ldstpd, int ldscld,
                const py::function fcn_f,
                const py::function fcn_fjacb,
                const py::function fcn_fjacd,
                py::array_t<double, py::array::c_style> beta,
                py::array_t<double, py::array::c_style> y,
                py::array_t<double, py::array::c_style> x,
                py::array_t<double, py::array::c_style> delta,
                std::optional<py::array_t<double, py::array::c_style>> we,
                std::optional<py::array_t<double, py::array::c_style>> wd,
                std::optional<py::array_t<int, py::array::c_style>> ifixb,
                std::optional<py::array_t<int, py::array::c_style>> ifixx,
                std::optional<py::array_t<double, py::array::c_style>> stpb,
                std::optional<py::array_t<double, py::array::c_style>> stpd,
                std::optional<py::array_t<double, py::array::c_style>> sclb,
                std::optional<py::array_t<double, py::array::c_style>> scld,
                std::optional<py::array_t<double, py::array::c_style>> lower,
                std::optional<py::array_t<double, py::array::c_style>> upper,
                std::optional<py::array_t<double, py::array::c_style>> work,
                std::optional<py::array_t<int, py::array::c_style>> iwork,
                std::optional<int> job, std::optional<int> ndigit,
                std::optional<double> taufac, std::optional<double> sstol,
                std::optional<double> partol, std::optional<int> maxit,
                std::optional<int> iprint, std::optional<std::string> errfile,
                std::optional<std::string> rptfile)

{
    // Create pointers to the NumPy arrays and scalar arguments
    // All input arrays are guaranteed to be contiguous and correctly shaped, allowing direct
    // pointer assignment without additional checks
    auto y_ptr = y.data();
    auto x_ptr = x.data();
    auto beta_ptr = beta.mutable_data();
    auto delta_ptr = delta.mutable_data();

    auto we_ptr = we ? we.value().data() : nullptr;
    auto wd_ptr = wd ? wd.value().data() : nullptr;
    auto ifixb_ptr = ifixb ? ifixb.value().data() : nullptr;
    auto ifixx_ptr = ifixx ? ifixx.value().data() : nullptr;

    auto stpb_ptr = stpb ? stpb.value().data() : nullptr;
    auto stpd_ptr = stpd ? stpd.value().data() : nullptr;
    auto sclb_ptr = sclb ? sclb.value().data() : nullptr;
    auto scld_ptr = scld ? scld.value().data() : nullptr;

    auto lower_ptr = lower ? lower.value().data() : nullptr;
    auto upper_ptr = upper ? upper.value().data() : nullptr;

    auto work_ptr = work ? work.value().mutable_data() : nullptr;
    auto iwork_ptr = iwork ? iwork.value().mutable_data() : nullptr;

    auto job_ptr = job ? &job.value() : nullptr;
    auto ndigit_ptr = ndigit ? &ndigit.value() : nullptr;
    auto taufac_ptr = taufac ? &taufac.value() : nullptr;
    auto sstol_ptr = sstol ? &sstol.value() : nullptr;
    auto partol_ptr = partol ? &partol.value() : nullptr;
    auto maxit_ptr = maxit ? &maxit.value() : nullptr;
    auto iprint_ptr = iprint ? &iprint.value() : nullptr;

    int lwork = 1;
    int liwork = 1;
    if (work) lwork = work.value().size();
    if (iwork) liwork = iwork.value().size();

    // Build static pointers to the Python functions
    // The static variables are necessary to ensure that the Python functions can be accessed
    // within the C-style function 'fcn'
    static py::function fcn_f_holder;
    fcn_f_holder = std::move(fcn_f);
    auto cleaner_1 = SelfCleaningPyObject(fcn_f_holder);

    static py::function fcn_fjacb_holder;
    fcn_fjacb_holder = std::move(fcn_fjacb);
    auto cleaner_2 = SelfCleaningPyObject(fcn_fjacb_holder);

    static py::function fcn_fjacd_holder;
    fcn_fjacd_holder = std::move(fcn_fjacd);
    auto cleaner_3 = SelfCleaningPyObject(fcn_fjacd_holder);

    static std::vector<ssize_t> shape_x{n};
    if (m != 1) shape_x.insert(shape_x.begin(), m);

    // Define the overall user-supplied model function 'fcn'
    odrpack_fcn_t fcn = nullptr;
    fcn = [](const int *n, const int *m, const int *npar, const int *nq,
             const int *ldn, const int *ldm, const int *ldnp, const double beta[],
             const double xplusd[], const int ifixb[], const int ifixx[],
             const int *ldifx, const int *ideval, double f[], double fjacb[],
             double fjacd[], int *istop) {
        *istop = 0;

        // Create NumPy arrays that wrap the input C-style arrays, without copying the data
        py::array_t<double> beta_ndarray(*npar, beta, py::none());
        py::array_t<double> xplusd_ndarray(shape_x, xplusd, py::none());

        try {
            // Evaluate model function
            if (*ideval % 10 > 0) {
                py::array_t<double, py::array::c_style> f_ndarray;
                f_ndarray = fcn_f_holder(beta_ndarray, xplusd_ndarray);
                const double *f_ndarray_ptr = f_ndarray.data();

                for (auto i = 0; i < (*nq) * (*ldn); i++) {
                    f[i] = f_ndarray_ptr[i];
                }
            }

            // Model partial derivatives wrt `beta`
            if ((*ideval / 10) % 10 != 0) {
                py::array_t<double, py::array::c_style> fjacb_ndarray;
                fjacb_ndarray = fcn_fjacb_holder(beta_ndarray, xplusd_ndarray);
                const double *fjacb_ndarray_ptr = fjacb_ndarray.data();

                for (auto i = 0; i < (*nq) * (*ldnp) * (*ldn); i++) {
                    fjacb[i] = fjacb_ndarray_ptr[i];
                }
            }

            // Model partial derivatives wrt `delta`
            if ((*ideval / 100) % 10 != 0) {
                py::array_t<double, py::array::c_style> fjacd_ndarray;
                fjacd_ndarray = fcn_fjacd_holder(beta_ndarray, xplusd_ndarray);
                const double *fjacd_ndarray_ptr = fjacd_ndarray.data();

                for (auto i = 0; i < (*nq) * (*ldnp) * (*ldn); i++) {
                    fjacd[i] = fjacd_ndarray_ptr[i];
                }
            }

        } catch (const py::error_already_set &e) {
            // temporary solution: need to figure out how to do this the right way
            std::string ewhat = e.what();
            if (ewhat.find("OdrStop") != std::string::npos) {
                std::cout << e.value() << std::endl;
                *istop = 1;
            } else {
                throw;
            }
        }
    };

    // Open files
    int lunrpt = 6;
    int lunerr = 6;
    int ierr = 1;

    if (rptfile) {
        lunrpt = 0;
        open_file(rptfile.value().c_str(), &lunrpt, &ierr);
        if (ierr != 0) throw std::runtime_error("Error opening report file.");
    }

    if (errfile) {
        if (errfile.value() != rptfile.value()) {
            lunerr = 0;
            open_file(errfile.value().c_str(), &lunerr, &ierr);
            if (ierr != 0) throw std::runtime_error("Error opening error file.");
        } else {
            lunerr = lunrpt;
        }
    }

    // Call the C function
    int info = -1;
    odr_long_c(fcn, &n, &m, &npar, &nq, &ldwe, &ld2we, &ldwd, &ld2wd, &ldifx,
               &ldstpd, &ldscld, &lwork, &liwork, beta_ptr, y_ptr, x_ptr, we_ptr,
               wd_ptr, ifixb_ptr, ifixx_ptr, stpb_ptr, stpd_ptr, sclb_ptr,
               scld_ptr, delta_ptr, lower_ptr, upper_ptr, work_ptr, iwork_ptr,
               job_ptr, ndigit_ptr, taufac_ptr, sstol_ptr, partol_ptr, maxit_ptr,
               iprint_ptr, &lunerr, &lunrpt, &info);

    // Close files
    if (rptfile) {
        close_file(&lunrpt, &ierr);
        if (ierr != 0) std::cerr << "Error closing report file." << std::endl;
    }

    if (errfile && lunrpt != lunerr) {
        close_file(&lunerr, &ierr);
        if (ierr != 0) std::cerr << "Error closing error file." << std::endl;
    }

    return info;
}

PYBIND11_MODULE(__odrpack, m) {
    m.def("odr", &odr_wrapper,
          R"doc(
C++ wrapper for the Orthogonal Distance Regression (ODR) routine.

Parameters
----------
n : int
    Number of observations.
m : int
    Number of columns in the independent variable data.
npar : int
    Number of function parameters.
nq : int
    Number of responses per observation.
ldwe : int
    Leading dimension of the `we` array, must be in `{1, n}`.
ld2we : int
    Second dimension of the `we` array, must be in `{1, nq}`.
ldwd : int
    Leading dimension of the `wd` array, must be in `{1, n}`.
ld2wd : int
    Second dimension of the `wd` array, must be in `{1, m}`.
ldifx : int
    Leading dimension of the `ifixx` array, must be in `{1, n}`.
ldstpd : int
    Leading dimension of the `stpd` array, must be in `{1, n}`.
ldscld : int
    Leading dimension of the `scld` array, must be in `{1, n}`.
f : Callable
    User-supplied function for evaluating the model, `f(beta, x)`.
fjacb : Callable
    User-supplied function for evaluating the Jacobian w.r.t. `beta`,
    `fjacb(beta, x)`.
fjacd : Callable
    User-supplied function for evaluating the Jacobian w.r.t. `delta`,
    `fjacd(beta, x)`.
beta : np.ndarray[float64]
    Array of function parameters with shape `(npar)`.
y : np.ndarray[float64]
    Dependent variables with shape `(nq, n)`. Ignored for implicit models.
x : np.ndarray[float64]
    Explanatory variables with shape `(m, n)`.
delta : np.ndarray[float64]
    Initial errors in `x` data with shape `(m, n)`.
we : np.ndarray[float64], optional
    Weights for `epsilon` with shape `(nq, ld2we, ldwe)`. Default is None.
wd : np.ndarray[float64], optional
    Weights for `delta` with shape `(m, ld2wd, ldwd)`. Default is None.
ifixb : np.ndarray[int32], optional
    Indicates fixed elements of `beta`. Default is None.
ifixx : np.ndarray[int32], optional
    Indicates fixed elements of `x`. Default is None.
stpb : np.ndarray[float64], optional
    Relative steps for finite difference derivatives w.r.t. `beta`. Default is None.
stpd : np.ndarray[float64], optional
    Relative steps for finite difference derivatives w.r.t. `delta`. Default is None.
sclb : np.ndarray[float64], optional
    Scaling values for `beta`. Default is None.
scld : np.ndarray[float64], optional
    Scaling values for `delta`. Default is None.
lower : np.ndarray[float64], optional
    Lower bounds for `beta`. Default is None.
upper : np.ndarray[float64], optional
    Upper bounds for `beta`. Default is None.
work : np.ndarray[float64], optional
    Real work space. Default is None.
iwork : np.ndarray[int32], optional
    Integer work space. Default is None.
job : int, optional
    Controls initialization and computational method. Default is None.
ndigit : int, optional
    Number of accurate digits in function results. Default is None.
taufac : float, optional
    Factor for initial trust region diameter. Default is None.
sstol : float, optional
    Sum-of-squares convergence tolerance. Default is None.
partol : float, optional
    Parameter convergence tolerance. Default is None.
maxit : int, optional
    Maximum number of iterations. Default is None.
iprint : int, optional
    Print control variable. Default is None.
errfile : str, optional
    Filename to use for error messages. Default is None.
rptfile : str, optional
    Filename to use for computation reports. Default is None.

Returns
-------
result : dict
    Dictionary with the following keys:
    - beta : np.ndarray[float64]
        Function parameters.
    - delta : np.ndarray[float64]
        Errors in `x` data.
    - work : np.ndarray[float64]
        Real work space.
    - iwork : np.ndarray[int32]
        Integer work space.
    - info : int
        Reason for stopping.

Notes
-----
- Ensure all array dimensions and functions are consistent with the provided arguments.
- Input arrays will automatically be made contiguous and cast to the correct type if necessary.
)doc",
          py::arg("n"), py::arg("m"), py::arg("npar"), py::arg("nq"),
          py::arg("ldwe"), py::arg("ld2we"), py::arg("ldwd"), py::arg("ld2wd"),
          py::arg("ldifx"), py::arg("ldstpd"), py::arg("ldscld"), py::arg("f"),
          py::arg("fjacb"), py::arg("fjacd"), py::arg("beta"), py::arg("y"),
          py::arg("x"),
          py::arg("delta"),
          py::arg("we").none(true) = nullptr,
          py::arg("wd").none(true) = nullptr,
          py::arg("ifixb").none(true) = nullptr,
          py::arg("ifixx").none(true) = nullptr,
          py::arg("stpb").none(true) = nullptr,
          py::arg("stpd").none(true) = nullptr,
          py::arg("sclb").none(true) = nullptr,
          py::arg("scld").none(true) = nullptr,
          py::arg("lower").none(true) = nullptr,
          py::arg("upper").none(true) = nullptr,
          py::arg("work").none(true) = nullptr,
          py::arg("iwork").none(true) = nullptr,
          py::arg("job").none(true) = nullptr,
          py::arg("ndigit").none(true) = nullptr,
          py::arg("taufac").none(true) = nullptr,
          py::arg("sstol").none(true) = nullptr,
          py::arg("partol").none(true) = nullptr,
          py::arg("maxit").none(true) = nullptr,
          py::arg("iprint").none(true) = nullptr,
          py::arg("errfile").none(true) = nullptr,
          py::arg("rptfile").none(true) = nullptr);

    // Calculate the dimensions of the workspace arrays
    m.def(
        "workspace_dimensions",
        [](int n, int m, int npar, int nq, bool isodr) {
            int lwork = 0;
            int liwork = 0;
            workspace_dimensions_c(&n, &m, &npar, &nq, &isodr, &lwork, &liwork);
            return std::make_tuple(lwork, liwork);
        },
        R"doc(
Calculate the dimensions of the workspace arrays.

Parameters
----------
n : int
    Number of observations.
m : int
    Number of columns of data in the explanatory variable.
npar : int
    Number of function parameters.
nq : int
    Number of responses per observation.
isodr : bool
    Variable designating whether the solution is by ODR (`True`) or by OLS (`False`).

Returns
-------
tuple
    A tuple containing the lengths of the work arrays (`lwork`, `liwork`).
)doc",
        py::arg("n"), py::arg("m"), py::arg("npar"), py::arg("nq"),
        py::arg("isodr").none(false));

    // Get storage locations within the integer work space
    m.def(
        "diwinf",
        [](int m, int npar, int nq) {
            iworkidx_t iworkidx = {};
            diwinf_c(&m, &npar, &nq, &iworkidx);
            return py::dict("msgb"_a = iworkidx.msgb,
                            "msgd"_a = iworkidx.msgd,
                            "ifix2"_a = iworkidx.ifix2,
                            "istop"_a = iworkidx.istop,
                            "nnzw"_a = iworkidx.nnzw,
                            "npp"_a = iworkidx.npp,
                            "idf"_a = iworkidx.idf,
                            "job"_a = iworkidx.job,
                            "iprin"_a = iworkidx.iprin,
                            "luner"_a = iworkidx.luner,
                            "lunrp"_a = iworkidx.lunrp,
                            "nrow"_a = iworkidx.nrow,
                            "ntol"_a = iworkidx.ntol,
                            "neta"_a = iworkidx.neta,
                            "maxit"_a = iworkidx.maxit,
                            "niter"_a = iworkidx.niter,
                            "nfev"_a = iworkidx.nfev,
                            "njev"_a = iworkidx.njev,
                            "int2"_a = iworkidx.int2,
                            "irank"_a = iworkidx.irank,
                            "ldtt"_a = iworkidx.ldtt,
                            "bound"_a = iworkidx.bound,
                            "liwkmn"_a = iworkidx.liwkmn);
        },
        R"doc(
Get storage locations within the integer work space.

Parameters
----------
m : int
    Number of columns of data in the explanatory variable.
npar : int
    Number of function parameters.
nq : int
    Number of responses per observation.

Returns
-------
dict
    A dictionary containing the 0-based indexes of the integer work array.
)doc",
        py::arg("m"), py::arg("npar"), py::arg("nq"));

    // Get storage locations within the real work space
    m.def(
        "dwinf",
        [](int n, int m, int npar, int nq, int ldwe, int ld2we, bool isodr) {
            workidx_t workidx = {};
            dwinf_c(&n, &m, &npar, &nq, &ldwe, &ld2we, &isodr, &workidx);
            return py::dict(
                "delta"_a = workidx.delta,
                "eps"_a = workidx.eps,
                "xplus"_a = workidx.xplus,
                "fn"_a = workidx.fn,
                "sd"_a = workidx.sd,
                "vcv"_a = workidx.vcv,
                "rvar"_a = workidx.rvar,
                "wss"_a = workidx.wss,
                "wssde"_a = workidx.wssde,
                "wssep"_a = workidx.wssep,
                "rcond"_a = workidx.rcond,
                "eta"_a = workidx.eta,
                "olmav"_a = workidx.olmav,
                "tau"_a = workidx.tau,
                "alpha"_a = workidx.alpha,
                "actrs"_a = workidx.actrs,
                "pnorm"_a = workidx.pnorm,
                "rnors"_a = workidx.rnors,
                "prers"_a = workidx.prers,
                "partl"_a = workidx.partl,
                "sstol"_a = workidx.sstol,
                "taufc"_a = workidx.taufc,
                "epsma"_a = workidx.epsma,
                "beta0"_a = workidx.beta0,
                "betac"_a = workidx.betac,
                "betas"_a = workidx.betas,
                "betan"_a = workidx.betan,
                "s"_a = workidx.s,
                "ss"_a = workidx.ss,
                "ssf"_a = workidx.ssf,
                "qraux"_a = workidx.qraux,
                "u"_a = workidx.u,
                "fs"_a = workidx.fs,
                "fjacb"_a = workidx.fjacb,
                "we1"_a = workidx.we1,
                "diff"_a = workidx.diff,
                "delts"_a = workidx.delts,
                "deltn"_a = workidx.deltn,
                "t"_a = workidx.t,
                "tt"_a = workidx.tt,
                "omega"_a = workidx.omega,
                "fjacd"_a = workidx.fjacd,
                "wrk1"_a = workidx.wrk1,
                "wrk2"_a = workidx.wrk2,
                "wrk3"_a = workidx.wrk3,
                "wrk4"_a = workidx.wrk4,
                "wrk5"_a = workidx.wrk5,
                "wrk6"_a = workidx.wrk6,
                "wrk7"_a = workidx.wrk7,
                "lower"_a = workidx.lower,
                "upper"_a = workidx.upper,
                "lwkmn"_a = workidx.lwkmn);
        },
        R"doc(
Get storage locations within the real work space.

Parameters
----------
n : int
    Number of observations.
m : int
    Number of columns of data in the explanatory variable.
npar : int
    Number of function parameters.
nq : int
    Number of responses per observation.
ldwe : int
    Leading dimension of the `we` array.
ld2we : int
    Second dimension of the `we` array.
isodr : bool
    Indicates whether the solution is by ODR (True) or by OLS (False).

Returns
-------
dict
    A dictionary containing the 0-based indexes of the real work array.
)doc",
        py::arg("n"), py::arg("m"), py::arg("npar"), py::arg("nq"),
        py::arg("ldwe"), py::arg("ld2we"), py::arg("isodr").none(false));
}