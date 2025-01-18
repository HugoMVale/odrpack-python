#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

#include "odrpack/odrpack.h"

namespace nb = nanobind;
using namespace nanobind::literals;

/*
The following class is necessary to ensure that the static variables used to store the callback
functions are automatically reset upon normal or abnormal exit from the `odr_wrapper` function.
From: https://github.com/libprima/prima/blob/main/python/_prima.cpp
*/
class SelfCleaningPyObject {
    nb::object &obj;

   public:
    SelfCleaningPyObject(nb::object &obj) : obj(obj) {}
    ~SelfCleaningPyObject() { obj = nb::none(); }
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
int odr_wrapper(int n,
                int m,
                int npar,
                int nq,
                int ldwe,
                int ld2we,
                int ldwd,
                int ld2wd,
                int ldifx,
                int ldstpd,
                int ldscld,
                const nb::callable fcn_f,
                const nb::callable fcn_fjacb,
                const nb::callable fcn_fjacd,
                nb::ndarray<double, nb::c_contig> beta,
                nb::ndarray<const double, nb::c_contig> y,
                nb::ndarray<const double, nb::c_contig> x,
                nb::ndarray<double, nb::c_contig> delta,
                std::optional<nb::ndarray<const double, nb::c_contig>> we,
                std::optional<nb::ndarray<const double, nb::c_contig>> wd,
                std::optional<nb::ndarray<const int, nb::c_contig>> ifixb,
                std::optional<nb::ndarray<const int, nb::c_contig>> ifixx,
                std::optional<nb::ndarray<const double, nb::c_contig>> stpb,
                std::optional<nb::ndarray<const double, nb::c_contig>> stpd,
                std::optional<nb::ndarray<const double, nb::c_contig>> sclb,
                std::optional<nb::ndarray<const double, nb::c_contig>> scld,
                std::optional<nb::ndarray<const double, nb::c_contig>> lower,
                std::optional<nb::ndarray<const double, nb::c_contig>> upper,
                std::optional<nb::ndarray<double, nb::c_contig>> work,
                std::optional<nb::ndarray<int, nb::c_contig>> iwork,
                std::optional<int> job,
                std::optional<int> ndigit,
                std::optional<double> taufac,
                std::optional<double> sstol,
                std::optional<double> partol,
                std::optional<int> maxit,
                std::optional<int> iprint,
                std::optional<std::string> errfile,
                std::optional<std::string> rptfile)

{
    // Create pointers to the NumPy arrays and scalar arguments
    // All input arrays are guaranteed to be contiguous and correctly shaped, allowing direct
    // pointer assignment without additional checks
    auto y_ptr = y.data();
    auto x_ptr = x.data();
    auto beta_ptr = beta.data();
    auto delta_ptr = delta.data();

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

    auto work_ptr = work ? work.value().data() : nullptr;
    auto iwork_ptr = iwork ? iwork.value().data() : nullptr;

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
    static nb::callable fcn_f_holder;
    fcn_f_holder = std::move(fcn_f);
    auto cleaner_1 = SelfCleaningPyObject(fcn_f_holder);

    static nb::callable fcn_fjacb_holder;
    fcn_fjacb_holder = std::move(fcn_fjacb);
    auto cleaner_2 = SelfCleaningPyObject(fcn_fjacb_holder);

    static nb::callable fcn_fjacd_holder;
    fcn_fjacd_holder = std::move(fcn_fjacd);
    auto cleaner_3 = SelfCleaningPyObject(fcn_fjacd_holder);

    // Define the overall user-supplied model function 'fcn'
    odrpack_fcn_t fcn = nullptr;
    fcn = [](const int *n, const int *m, const int *npar, const int *nq,
             const int *ldn, const int *ldm, const int *ldnp, const double beta[],
             const double xplusd[], const int ifixb[], const int ifixx[],
             const int *ldifx, const int *ideval, double f[], double fjacb[],
             double fjacd[], int *istop) {
        // Create NumPy arrays that wrap the input C-style arrays, without copying the data
        nb::ndarray<const double, nb::numpy> beta_ndarray(beta, {static_cast<size_t>(*npar)});
        nb::ndarray<const double, nb::numpy> xplusd_ndarray(
            xplusd,
            (*m == 1) ? std::initializer_list<size_t>{static_cast<size_t>(*n)}
                      : std::initializer_list<size_t>{static_cast<size_t>(*m), static_cast<size_t>(*n)});

        *istop = 0;
        try {
            // Evaluate model function
            if (*ideval % 10 > 0) {
                auto f_object = fcn_f_holder(beta_ndarray, xplusd_ndarray);
                auto f_ndarray = nb::cast<nb::ndarray<const double, nb::c_contig>>(f_object);
                auto f_ndarray_ptr = f_ndarray.data();
                for (auto i = 0; i < (*nq) * (*ldn); i++) {
                    f[i] = f_ndarray_ptr[i];
                }
            }

            // Model partial derivatives wrt `beta`
            if ((*ideval / 10) % 10 != 0) {
                auto fjacb_object = fcn_fjacb_holder(beta_ndarray, xplusd_ndarray);
                auto fjacb_ndarray = nb::cast<nb::ndarray<const double, nb::c_contig>>(fjacb_object);
                auto fjacb_ndarray_ptr = fjacb_ndarray.data();
                for (auto i = 0; i < (*nq) * (*ldnp) * (*ldn); i++) {
                    fjacb[i] = fjacb_ndarray_ptr[i];
                }
            }

            // Model partial derivatives wrt `delta`
            if ((*ideval / 100) % 10 != 0) {
                auto fjacd_object = fcn_fjacd_holder(beta_ndarray, xplusd_ndarray);
                auto fjacd_ndarray = nb::cast<nb::ndarray<const double, nb::c_contig>>(fjacd_object);
                auto fjacd_ndarray_ptr = fjacd_ndarray.data();
                for (auto i = 0; i < (*nq) * (*ldnp) * (*ldn); i++) {
                    fjacd[i] = fjacd_ndarray_ptr[i];
                }
            }

        } catch (const nb::python_error &e) {
            // temporary solution: need to figure out how to do this the right way
            std::string ewhat = e.what();
            if (ewhat.find("OdrStop") != std::string::npos) {
                std::cerr << ewhat << std::endl;
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
        if (!((rptfile) && (errfile.value() == rptfile.value()))) {
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

NB_MODULE(__odrpack, m) {
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
info : int
    Reason for stopping.

Notes
-----
- Ensure all array dimensions and functions are consistent with the provided arguments.
- Input arrays will automatically be made contiguous and cast to the correct type if necessary.
    )doc",
          nb::arg("n"), nb::arg("m"), nb::arg("npar"), nb::arg("nq"),
          nb::arg("ldwe"), nb::arg("ld2we"), nb::arg("ldwd"), nb::arg("ld2wd"),
          nb::arg("ldifx"), nb::arg("ldstpd"), nb::arg("ldscld"),
          nb::arg("f"),
          nb::arg("fjacb"),
          nb::arg("fjacd"),
          nb::arg("beta"),
          nb::arg("y"),
          nb::arg("x"),
          nb::arg("delta"),
          nb::arg("we").none() = nullptr,
          nb::arg("wd").none() = nullptr,
          nb::arg("ifixb").none() = nullptr,
          nb::arg("ifixx").none() = nullptr,
          nb::arg("stpb").none() = nullptr,
          nb::arg("stpd").none() = nullptr,
          nb::arg("sclb").none() = nullptr,
          nb::arg("scld").none() = nullptr,
          nb::arg("lower").none() = nullptr,
          nb::arg("upper").none() = nullptr,
          nb::arg("work").none() = nullptr,
          nb::arg("iwork").none() = nullptr,
          nb::arg("job").none() = nullptr,
          nb::arg("ndigit").none() = nullptr,
          nb::arg("taufac").none() = nullptr,
          nb::arg("sstol").none() = nullptr,
          nb::arg("partol").none() = nullptr,
          nb::arg("maxit").none() = nullptr,
          nb::arg("iprint").none() = nullptr,
          nb::arg("errfile").none() = nullptr,
          nb::arg("rptfile").none() = nullptr);

    // Calculate the dimensions of the workspace arrays
    m.def(
        "workspace_dimensions",
        [](int n, int m, int npar, int nq, bool isodr) {
            int lwork = 0;
            int liwork = 0;
            workspace_dimensions_c(&n, &m, &npar, &nq, &isodr, &lwork, &liwork);
            return nb::make_tuple(lwork, liwork);
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
tuple[int, int]
    A tuple containing the lengths of the work arrays (`lwork`, `liwork`).
)doc",
        nb::arg("n"), nb::arg("m"), nb::arg("npar"), nb::arg("nq"),
        nb::arg("isodr"));

    // Get storage locations within the integer work space
    m.def(
        "diwinf",
        [](int m, int npar, int nq) {
            iworkidx_t iworkidx = {};
            diwinf_c(&m, &npar, &nq, &iworkidx);
            std::map<std::string, int> result;
            result["msgb"] = iworkidx.msgb;
            result["msgd"] = iworkidx.msgd;
            result["ifix2"] = iworkidx.ifix2;
            result["istop"] = iworkidx.istop;
            result["nnzw"] = iworkidx.nnzw;
            result["npp"] = iworkidx.npp;
            result["idf"] = iworkidx.idf;
            result["job"] = iworkidx.job;
            result["iprin"] = iworkidx.iprin;
            result["luner"] = iworkidx.luner;
            result["lunrp"] = iworkidx.lunrp;
            result["nrow"] = iworkidx.nrow;
            result["ntol"] = iworkidx.ntol;
            result["neta"] = iworkidx.neta;
            result["maxit"] = iworkidx.maxit;
            result["niter"] = iworkidx.niter;
            result["nfev"] = iworkidx.nfev;
            result["njev"] = iworkidx.njev;
            result["int2"] = iworkidx.int2;
            result["irank"] = iworkidx.irank;
            result["ldtt"] = iworkidx.ldtt;
            result["bound"] = iworkidx.bound;
            result["liwkmn"] = iworkidx.liwkmn;
            return result;
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
dict[str, int]
    A dictionary containing the 0-based indexes of the integer work array.
)doc",
        nb::arg("m"), nb::arg("npar"), nb::arg("nq"));

    // Get storage locations within the real work space
    m.def(
        "dwinf",
        [](int n, int m, int npar, int nq, int ldwe, int ld2we, bool isodr) {
            workidx_t workidx = {};
            dwinf_c(&n, &m, &npar, &nq, &ldwe, &ld2we, &isodr, &workidx);
            std::map<std::string, int> result;
            result["delta"] = workidx.delta;
            result["eps"] = workidx.eps;
            result["xplus"] = workidx.xplus;
            result["fn"] = workidx.fn;
            result["sd"] = workidx.sd;
            result["vcv"] = workidx.vcv;
            result["rvar"] = workidx.rvar;
            result["wss"] = workidx.wss;
            result["wssde"] = workidx.wssde;
            result["wssep"] = workidx.wssep;
            result["rcond"] = workidx.rcond;
            result["eta"] = workidx.eta;
            result["olmav"] = workidx.olmav;
            result["tau"] = workidx.tau;
            result["alpha"] = workidx.alpha;
            result["actrs"] = workidx.actrs;
            result["pnorm"] = workidx.pnorm;
            result["rnors"] = workidx.rnors;
            result["prers"] = workidx.prers;
            result["partl"] = workidx.partl;
            result["sstol"] = workidx.sstol;
            result["taufc"] = workidx.taufc;
            result["epsma"] = workidx.epsma;
            result["beta0"] = workidx.beta0;
            result["betac"] = workidx.betac;
            result["betas"] = workidx.betas;
            result["betan"] = workidx.betan;
            result["s"] = workidx.s;
            result["ss"] = workidx.ss;
            result["ssf"] = workidx.ssf;
            result["qraux"] = workidx.qraux;
            result["u"] = workidx.u;
            result["fs"] = workidx.fs;
            result["fjacb"] = workidx.fjacb;
            result["we1"] = workidx.we1;
            result["diff"] = workidx.diff;
            result["delts"] = workidx.delts;
            result["deltn"] = workidx.deltn;
            result["t"] = workidx.t;
            result["tt"] = workidx.tt;
            result["omega"] = workidx.omega;
            result["fjacd"] = workidx.fjacd;
            result["wrk1"] = workidx.wrk1;
            result["wrk2"] = workidx.wrk2;
            result["wrk3"] = workidx.wrk3;
            result["wrk4"] = workidx.wrk4;
            result["wrk5"] = workidx.wrk5;
            result["wrk6"] = workidx.wrk6;
            result["wrk7"] = workidx.wrk7;
            result["lower"] = workidx.lower;
            result["upper"] = workidx.upper;
            result["lwkmn"] = workidx.lwkmn;
            return result;
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
dict[str, int]
    A dictionary containing the 0-based indexes of the real work array.
    )doc",
        nb::arg("n"), nb::arg("m"), nb::arg("npar"), nb::arg("nq"),
        nb::arg("ldwe"), nb::arg("ld2we"), nb::arg("isodr"));
}