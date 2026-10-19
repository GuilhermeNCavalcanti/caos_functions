#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatically converts std::vector to Python lists
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>

// Estado r(t) com tamanho fixo
struct State3D {
    double x;
    double y;
    double z;
};

// Armazena os Eventos
struct CrossTracker {
    std::vector<double>* cross_times;
    std::vector<double>* cross_x;
    std::vector<double>* cross_y;
    std::vector<double>* cross_z;
    int max_cross;
    int total_cross;
    double start_time;
};

// Parâmetros do sistema
struct SystemParams {
    double sigma; 
    double beta; 
    double rho;
};

// Armazena os estados em cada passo aceito
struct TrajectoryTracker {
    std::vector<double> t;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<double> dz;
};

// Informações da Integração
struct Status {
    int nAccepted = 0;
    int nRejected = 0;
    int varNum;
    int dir;
    double crossVal;
};

// Butcher Tableau para o DOP853
namespace DOP853Const {
    constexpr int STAGES = 12;
    constexpr double C[STAGES] = {
        0.0,
        0.526001519587677318785587544488e-01,
        0.789002279381515978178381316732e-01,
        0.118350341907227396726757197510,
        0.281649658092772603273242802490,
        0.333333333333333333333333333333,
        0.25,
        0.307692307692307692307692307692,
        0.651282051282051282051282051282,
        0.6,
        0.857142857142857142857142857142,
        1.0,
    };
    constexpr double B[STAGES] = {
        5.42937341165687622380535766363e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        4.45031289275240888144113950566,
        1.89151789931450038304281599044,
        -5.8012039600105847814672114227,
        3.1116436695781989440891606237e-1,
        -1.52160949662516078556178806805e-1,
        2.01365400804030348374776537501e-1,
        4.47106157277725905176885569043e-2,
    };
    constexpr double ER[STAGES] = {
        0.1312004499419488073250102996e-01,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.1225156446376204440720569753e+01,
        -0.4957589496572501915214079952,
        0.1664377182454986536961530415e+01,
        -0.3503288487499736816886487290,
        0.3341791187130174790297318841,
        0.8192320648511571246570742613e-01,
        -0.2235530786388629525884427845e-01,
    };
    constexpr double A[STAGES][STAGES] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {5.26001519587677318785587544488e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.97250569845378994544595329183e-2, 5.91751709536136983633785987549e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.95875854768068491816892993775e-2, 0.0, 8.87627564304205475450678981324e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.41365134159266685502369798665e-1, 0.0, -8.84549479328286085344864962717e-1, 9.24834003261792003115737966543e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.7037037037037037037037037037e-2, 0.0, 0.0, 1.70828608729473871279604482173e-1, 1.25467687566822425016691814123e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.7109375e-2, 0.0, 0.0, 1.70252211019544039314978060272e-1, 6.02165389804559606850219397283e-2, -1.7578125e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.70920001185047927108779319836e-2, 0.0, 0.0, 1.70383925712239993810214054705e-1, 1.07262030446373284651809199168e-1, -1.53194377486244017527936158236e-2, 8.27378916381402288758473766002e-3, 0.0, 0.0, 0.0, 0.0, 0.0},
        {6.24110958716075717114429577812e-1, 0.0, 0.0, -3.36089262944694129406857109825, -8.68219346841726006818189891453e-1, 2.75920996994467083049415600797e+1, 2.01540675504778934086186788979e+1, -4.34898841810699588477366255144e+1, 0.0, 0.0, 0.0, 0.0},
        {4.77662536438264365890433908527e-1, 0.0, 0.0, -2.48811461997166764192642586468, -5.90290826836842996371446475743e-1, 2.12300514481811942347288949897e+1, 1.52792336328824235832596922938e+1, -3.32882109689848629194453265587e+1, -2.03312017085086261358222928593e-2, 0.0, 0.0, 0.0},
        {-9.3714243008598732571704021658e-1, 0.0, 0.0, 5.18637242884406370830023853209, 1.09143734899672957818500254654, -8.14978701074692612513997267357, -1.85200656599969598641566180701e+1, 2.27394870993505042818970056734e+1, 2.49360555267965238987089396762, -3.0467644718982195003823669022, 0.0, 0.0},
        {2.27331014751653820792359768449, 0.0, 0.0, -1.05344954667372501984066689879e+1, -2.00087205822486249909675718444, -1.79589318631187989172765950534e+1, 2.79488845294199600508499808837e+1, -2.85899827713502369474065508674, -8.87285693353062954433549289258, 1.23605671757943030647266201528e+1, 6.43392746015763530355970484046e-1, 0.0},
    };
}

// Sistema de Lorenz r'
inline void lorenzODE(
    double t, 
    const State3D& r, // State r(t) 
    State3D& drdt, // State r'(t) a ser determinado
    const SystemParams& params) {
        drdt.x = params.sigma * (r.y - r.x);
        drdt.y = r.x * (params.rho - r.z) - r.y;
        drdt.z = r.x * r.y - params.beta * r.z;
    }

// Derivadas do sistema r''
inline void lorenzODE_dt2(
    double t, 
    const State3D& r, // State r(t)
    const State3D& drdt, // State r'(t)
    State3D& dr2dt2, // State r''(t) a ser determinado
    const SystemParams& params) {
        dr2dt2.x = params.sigma * (drdt.y - drdt.x);
        dr2dt2.y = drdt.x * (params.rho - r.z) - r.x * drdt.z - drdt.y;
        dr2dt2.z = drdt.x * r.y + r.x * drdt.y - params.beta * drdt.z;
    }

// Core Step Function
bool dop853_step(
    double& t, State3D& r, 
    double& h, // Time step
    double rtol, double atol, // Error tolerances
    const SystemParams& params) {
        State3D k[DOP853Const::STAGES]; // Array with STAGES (12 for DOP853) states to perform step
        State3D r_temp; // Auxiliar state

        // Stage 1
        lorenzODE(t, r, k[0], params);

        // Stage 2 to 12
        for (int i = 1; i < DOP853Const::STAGES; ++i) {
            r_temp.x = r.x;
            r_temp.y = r.y;
            r_temp.z = r.z;

            for (int j = 0; j < i; ++j) {
                r_temp.x += h * DOP853Const::A[i][j] * k[j].x;
                r_temp.y += h * DOP853Const::A[i][j] * k[j].y; 
                r_temp.z += h * DOP853Const::A[i][j] * k[j].z; 
            }
            lorenzODE(t + h * DOP853Const::C[i], r_temp, k[i], params);
        }

        // Combine stages for candidate solution and error estimation
        State3D r_next = r;
        State3D error = {0.0, 0.0, 0.0};

        for (int i = 0; i < DOP853Const::STAGES; ++i) {
            r_next.x += h * DOP853Const::B[i] * k[i].x;
            r_next.y += h * DOP853Const::B[i] * k[i].y;
            r_next.z += h * DOP853Const::B[i] * k[i].z;

            error.x += h * DOP853Const::ER[i] * k[i].x;
            error.y += h * DOP853Const::ER[i] * k[i].y;
            error.z += h * DOP853Const::ER[i] * k[i].z;
        }

        // Adaptive step size control
        double scale_x = atol + rtol * fmax(fabs(r.x), fabs(r_next.x));
        double scale_y = atol + rtol * fmax(fabs(r.y), fabs(r_next.y));
        double scale_z = atol + rtol * fmax(fabs(r.z), fabs(r_next.z));

        double err_x = fabs(error.x) / scale_x;
        double err_y = fabs(error.y) / scale_y;
        double err_z = fabs(error.z) / scale_z;

        // RMS error across the 3 dimensions
        double max_err = sqrt((1.0/3.0) * (err_x * err_x + err_y * err_y + err_z * err_z));

        if (max_err <= 1.0) {
            // Accept Step
            t += h; // Update the current time
            r = r_next; // Update the state
        }

        // Compute next step size
        double factor = 0.9 * pow(1.0 / fmax(max_err, 1e-10), 1.0/8.0);
        factor = fmin(5.0, fmax(0.1, factor)); // Clamp factor
        h *= factor; // Update next step
        
        // Returns if the step was accepted (true) or rejected (false)
        return (max_err <= 1.0);
}

// Cubic Spline 
double cubicSpline(
    double s, double h, 
    double x1, double dx1,
    double x2, double dx2){
    double s2 = s * s;
    double s3 = s2 * s;
    double spline = (2.0 * s3 - 3.0 * s2 + 1.0) * x1 +
                    (s3 - 2.0 * s2 + s) * h * dx1 +
                    (-2.0 * s3 + 3.0 * s2) * x2 + 
                    (s3 - s2) * h * dx2;
    return spline;
}

// Derivada do Cubic Spline (dx/ds)
double cubicSplineDerivative(
    double s, double h, 
    double x1, double dx1,
    double x2, double dx2){
    double s2 = s * s;
    double dx_ds = (6.0 * s2 - 6.0 * s) * x1 + 
                   (3.0 * s2 - 4.0 * s + 1.0) * h * dx1 + 
                   (-6.0 * s2 + 6.0 * s) * x2 + 
                   (3.0 * s2 - 2.0 * s) * h * dx2;
    return dx_ds;
}

// Interpola o tempo e o estado do Evento entre dois passos
template <int VARNUM>
void interpolateCrossing(
    double tNew, State3D rNew, State3D drNew, // Estado após o passo
    double tOld, State3D rOld, State3D drOld, // Estado antes do passo
    double hTaken, // Tamanho do passo
    double timeOffSet, // Offset do sub-intervalo atual
    double crossVal, // Valor de busca do evento
    const SystemParams& params, 
    CrossTracker& tracker){
        // r' para a interpolção de r
        // State3D drOld, drNew;
        // lorenzODE(tOld, rOld, drOld, params);
        // lorenzODE(tNew, rNew, drNew, params);

        // r'' para a interpolção de r'
        State3D dr2Old, dr2New;
        lorenzODE_dt2(tOld, rOld, drOld, dr2Old, params);
        lorenzODE_dt2(tNew, rNew, drNew, dr2New, params);

        double xInterpOld, dxInterpOld;
        double xInterpNew, dxInterpNew;
        if constexpr (VARNUM == 1) { // Evento em x
            xInterpOld = rOld.x;
            dxInterpOld = drOld.x;
            xInterpNew = rNew.x;
            dxInterpNew = drNew.x;
        } else if constexpr (VARNUM == 2) { // Evento em y
            xInterpOld = rOld.y;
            dxInterpOld = drOld.y;
            xInterpNew = rNew.y;
            dxInterpNew = drNew.y;
        } else if constexpr (VARNUM == 3) { // Evento em z
            xInterpOld = rOld.z;
            dxInterpOld = drOld.z;
            xInterpNew = rNew.z;
            dxInterpNew = drNew.z;
        } else if constexpr (VARNUM == -1) { // Evento em x'
            xInterpOld = drOld.x;
            dxInterpOld = dr2Old.x;
            xInterpNew = drNew.x;
            dxInterpNew = dr2New.x;
        } else if constexpr (VARNUM == -2) { // Evento em y'
            xInterpOld = drOld.y;
            dxInterpOld = dr2Old.y;
            xInterpNew = drNew.y;
            dxInterpNew = dr2New.y;
        } else if constexpr (VARNUM == -3) { // Evento em z'
            xInterpOld = drOld.z;
            dxInterpOld = dr2Old.z;
            xInterpNew = drNew.z;
            dxInterpNew = dr2New.z;
        }

        // Initial guess for the root fraction 's' (Linear Interpolation)
        double s = (crossVal - xInterpOld) / (xInterpNew - xInterpOld);

        // Newton-Raphson Iteration to determine 's'
        for (int iter = 0; iter < 25; ++iter) {

            // Evaluate the Cubic dx Spline
            double dx_Spline = cubicSpline(s, hTaken, xInterpOld, dxInterpOld, xInterpNew, dxInterpNew);

            // Evaluate the Derivative v'(s) = dv/ds
            double dx_ds_Spline = cubicSplineDerivative(s, hTaken, xInterpOld, dxInterpOld, xInterpNew, dxInterpNew);

            // Break if it hits a flat slope to prevent division by zero
            if (fabs(dx_ds_Spline) < 1e-15) break;

            // Newton-Raphson update
            double ds = (dx_Spline - crossVal) / dx_ds_Spline;
            s -= ds;

            // Break if converged
            if (fabs(ds) < 1e-14) break;
        }

        // Clamp 's' strictly between 0 and 1 to guarantee it stays inside the step
        s = fmax(0.0, fmin(1.0, s));

        // Interpolate tCross with the refined 's'
        double tCross = tOld + hTaken * s + timeOffSet; // In GLOBAL time

        // Check if the crossing should be tracked
        if (tCross >= tracker.start_time) {
            // Interpolate rCross using the Cubic Position Spline
            double xCross = cubicSpline(s, hTaken, rOld.x, drOld.x, rNew.x, drNew.x);
            double yCross = cubicSpline(s, hTaken, rOld.y, drOld.y, rNew.y, drNew.y);
            double zCross = cubicSpline(s, hTaken, rOld.z, drOld.z, rNew.z, drNew.z);
            // State3D rCross = {xCross, yCross, zCross};
            // If the number of crossings is unbound
            if (tracker.max_cross < 0) {
                tracker.cross_times -> push_back(tCross);
                tracker.cross_x -> push_back(xCross);
                tracker.cross_y -> push_back(yCross);
                tracker.cross_z -> push_back(zCross);
                // tracker.cross_positions -> push_back(rCross);
            } else { 
                // If only max_cross should be tracked
                int index = tracker.total_cross % tracker.max_cross;
                (*tracker.cross_times)[index] = tCross;
                (*tracker.cross_x)[index] = xCross;
                (*tracker.cross_y)[index] = yCross;
                (*tracker.cross_z)[index] = zCross;
                // (*tracker.cross_positions)[index] = rCross;
            }
            tracker.total_cross++;
        }
}

// Loop de Integração
template <int VARNUM, int DIR>
void integrate_dop853(
    State3D& r, double& t,
    const double* breakpoints, int num_breakpoints, // Sub-intervals time limits
    double rtol, double atol, double initialStep, // Error tolerances and initial step size
    const SystemParams& params, // Parameters that defines the system
    CrossTracker& crossTracker, double crossVal,
    TrajectoryTracker& trajectory, // Tracker of validated steps
    Status& integrationStatus){ 
        // Record the initial state using the GLOBAL initial time
        State3D drIni;
        lorenzODE(t, r, drIni, params);
        trajectory.t.push_back(breakpoints[0]);
        trajectory.x.push_back(r.x);
        trajectory.y.push_back(r.y);
        trajectory.z.push_back(r.z);
        trajectory.dx.push_back(drIni.x);
        trajectory.dy.push_back(drIni.y);
        trajectory.dz.push_back(drIni.z);

        // Perform the integration along each sub-interval
        for (int b = 1; b < num_breakpoints; ++b) {
            // Local size of this sub-interval
            double target_t = breakpoints[b] - breakpoints[b-1];

            if (target_t <= 0) continue;

            double h = initialStep; // Initial step guess

            // Integrate purely in local time [0, target_t]
            while (t < target_t) {
                if (t + h > target_t) h = target_t - t;
                
                // State before next step
                double tOld = t;
                State3D rOld = r;
                State3D drOld;
                lorenzODE(tOld, rOld, drOld, params);

                // Size of the next step
                double h_taken = h;
    
                bool accepted = dop853_step(t, r, h, rtol, atol, params);
                if (accepted) {
                    integrationStatus.nAccepted++;
                    // Determine the offset for this specific sub-interval
                    double current_offset = breakpoints[b-1];
                    State3D drNew;
                    lorenzODE(t, r, drNew, params);
                    // Verify if the step crosses the startTime of the trackers
                    if (t + current_offset >= crossTracker.start_time) { 
                        // Verify if r or r' crossed the target plane
                        if constexpr (VARNUM == 1) { // Look for crosses in the x variable
                            if constexpr (DIR == -1) {
                                if (rOld.x > crossVal && r.x <= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            } else if constexpr (DIR == 1){
                                if (rOld.x < crossVal && r.x >= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            }
                        } else if constexpr (VARNUM == 2) { // Look for crosses in the y variable
                            if constexpr (DIR == -1) {
                                if (rOld.y > crossVal && r.y <= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            } else if constexpr (DIR == 1){
                                if (rOld.y < crossVal && r.y >= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            }
                        } else if constexpr (VARNUM == 3) { // Look for crosses in the z variable
                            if constexpr (DIR == -1) {
                                if (rOld.z > crossVal && r.z <= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            } else if constexpr (DIR == 1){
                                if (rOld.z < crossVal && r.z >= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            }
                        } else if constexpr (VARNUM == -1) { // Look for crosses in the x' variable
                            if constexpr (DIR == -1) {
                                if (drOld.x > crossVal && drNew.x <= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            } else if constexpr (DIR == 1){
                                if (drOld.x < crossVal && drNew.x >= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            }
                        } else if constexpr (VARNUM == -2) { // Look for crosses in the y' variable
                            if constexpr (DIR == -1) {
                                if (drOld.y > crossVal && drNew.y <= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            } else if constexpr (DIR == 1){
                                if (drOld.y < crossVal && drNew.y >= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            }
                        } else if constexpr (VARNUM == -3) { // Look for crosses in the z' variable
                            if constexpr (DIR == -1) {
                                if (drOld.z > crossVal && drNew.z <= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            } else if constexpr (DIR == 1){
                                if (drOld.z < crossVal && drNew.z >= crossVal) interpolateCrossing<VARNUM>(t, r, drNew, tOld, rOld, drOld, h_taken, current_offset, crossVal, params, crossTracker);
                            }
                        }
                    }

                    // Add the accepted step to the tracker
                    trajectory.t.push_back(t + current_offset);
                    trajectory.x.push_back(r.x);
                    trajectory.y.push_back(r.y);
                    trajectory.z.push_back(r.z);
                    trajectory.dx.push_back(drNew.x);
                    trajectory.dy.push_back(drNew.y);
                    trajectory.dz.push_back(drNew.z);

                    // Interrupt the integration if all desired crosses were found
                    if (crossTracker.total_cross == crossTracker.max_cross) return;
                }
                else integrationStatus.nRejected++;
            }
            // Reset local time to 0 for the start of the next sub-interval
            t = 0.0;
        }
    }

namespace py = pybind11;

template <typename T>
// Helper function to convert std::vector<T> to py::array_t<T>, allowing to return NumPy arrays directly from C++ without copying the data
py::array_t<T> as_pyarray(std::vector<T>&& vec) {
    // Move the vector to the heap so it survives after the function returns
    auto* ptr = new std::vector<T>(std::move(vec));

    // Create a Python capsule to delete the vector later
    auto capsule = py::capsule(ptr, [](void* p) {
        delete reinterpret_cast<std::vector<T>*>(p);
    });

    // Return the NumPy array pointing directly to the C++ memory
    return py::array_t<T>(ptr->size(), ptr->data(), capsule);
}

// Wrapper function Python will actually call
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, 
           py::array_t<double>, py::array_t<double>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, 
           std::tuple<int, int, int, int, double>>
run_simulation(std::vector<double> r0,
               std::vector<double> breakpoints,
               double sigma, double beta, double rho, 
               int varNum, int dir, double crossVal,
               int max_cross, double trackerStartTime, 
               double rtol, double atol, double initialStep)
{
    // Setup the inital state
    State3D r = {r0.at(0), r0.at(1), r0.at(2)};
    // Start local integration time at 0
    double t = 0.0;
    // Sets the initial system parameters

    SystemParams params = {sigma, beta, rho};

    double totalIntegrationTime = breakpoints.back() - breakpoints.front();

    // Estimate total steps with 10 steps per time unit and a minimum of 1000
    int estimatedSteps = std::max(1000, static_cast<int>(10.0 * totalIntegrationTime));

    // Creates the steps tracker
    TrajectoryTracker trajectory;
    // Pre-allocate memory to minimize dynamic reallocation 
    trajectory.t.reserve(estimatedSteps); 
    trajectory.x.reserve(estimatedSteps); 
    trajectory.y.reserve(estimatedSteps); 
    trajectory.z.reserve(estimatedSteps);

    // Dynamically allocate memory for the number of peaks requested
    int initialSize = (max_cross > 0) ? max_cross : 0;
    std::vector<double> cross_t_mem(initialSize, NAN);
    std::vector<double> cross_x_mem(initialSize, NAN);
    std::vector<double> cross_y_mem(initialSize, NAN);
    std::vector<double> cross_z_mem(initialSize, NAN);

    // Point the tracker at this memory
    CrossTracker trackerCrossings = {&cross_t_mem, &cross_x_mem, &cross_y_mem, &cross_z_mem, max_cross, 0, trackerStartTime};

    // Get the raw pointer and size to pass to the core function
    const double* bp_ptr = breakpoints.data();
    // The number of breakpoints never reach billions
    int num_bp = static_cast<int>(breakpoints.size());

    Status integrationStatus = {0, 0, varNum, dir, crossVal};

    // Release Python GIL before starting the C++ integration Loop
    py::gil_scoped_release release;

    // Perform the integration
    if (varNum == 1) {
        if (dir == 1) integrate_dop853<1, 1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
        else if (dir == -1) integrate_dop853<1, -1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
    } else if (varNum == 2) {
        if (dir == 1) integrate_dop853<2, 1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
        else if (dir == -1) integrate_dop853<2, -1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
    } else if (varNum == 3) {
        if (dir == 1) integrate_dop853<3, 1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
        else if (dir == -1) integrate_dop853<3, -1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
    } else if (varNum == -1) {
        if (dir == 1) integrate_dop853<-1, 1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
        else if (dir == -1) integrate_dop853<-1, -1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
    } else if (varNum == -2) {
        if (dir == 1) integrate_dop853<-2, 1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
        else if (dir == -1) integrate_dop853<-2, -1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
    } else if (varNum == -3) {
        if (dir == 1) integrate_dop853<-3, 1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
        else if (dir == -1) integrate_dop853<-3, -1>(r, t, bp_ptr, num_bp, rtol, atol, initialStep, params, trackerCrossings, crossVal, trajectory, integrationStatus);
    }

    // Re-acquire the Python GIL after performing the integration and before before using the Python C-API
    py::gil_scoped_acquire acquire;

    // Package the results for Python
    std::vector<double> py_cross_times;
    std::vector<double> py_cross_x, py_cross_y, py_cross_z;
    py_cross_times = cross_t_mem;
    py_cross_x = cross_x_mem;
    py_cross_y = cross_y_mem;
    py_cross_z = cross_z_mem;

    // Return the Trajectory and Peaks as NumPy arrays and a tuple with Status
    return std::make_tuple(
        as_pyarray(std::move(trajectory.t)),
        as_pyarray(std::move(trajectory.x)),
        as_pyarray(std::move(trajectory.y)),
        as_pyarray(std::move(trajectory.z)),
        as_pyarray(std::move(trajectory.dx)),
        as_pyarray(std::move(trajectory.dy)),
        as_pyarray(std::move(trajectory.dz)),
        as_pyarray(std::move(py_cross_times)),
        as_pyarray(std::move(py_cross_x)),
        as_pyarray(std::move(py_cross_y)),
        as_pyarray(std::move(py_cross_z)),
        std::make_tuple(integrationStatus.nAccepted, integrationStatus.nRejected, integrationStatus.varNum, integrationStatus.dir, integrationStatus.crossVal)
    );
}

// Create the Python module using pybind11
PYBIND11_MODULE(lorenz_cpp, m) {
    m.doc() = "C++ DOP853 Integrator for Lorenz System";
    m.def("run_simulation", &run_simulation, 
          py::arg("r0"),
          py::arg("breakpoints"),
          py::arg("sigma"), py::arg("beta"), py::arg("rho"),
          py::arg("varNum"), py::arg("dir"), py::arg("crossVal"),
          py::arg("max_cross") = 100, py::arg("trackerStartTime") = 0.0,
          py::arg("rtol") = 1e-7, py::arg("atol") = 1e-10, py::arg("initialStep") = 1e-6);
}