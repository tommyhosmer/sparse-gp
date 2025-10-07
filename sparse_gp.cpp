#include <stdlib.h>
#include <utility>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense"
#include "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/unsupported/Eigen/MatrixFunctions"
#include <cassert>


using namespace Eigen;


class SparseGP1D {
    int n;
    int m;
    double sigma_y;

public:
    SparseGP1D(int n, int m, double sigma_y) : n(n), m(m), sigma_y(sigma_y) {}
    ~SparseGP1D() {}

    VectorXd func(const VectorXd& x) {
        // Latent function 
        return 1.0 * (3 * x.array() * M_PI).sin() + 
               0.3 * (x.array() * 9 * M_PI).cos() + 
               0.5 * (x.array() * 7 * M_PI).sin();
    }

    std::vector<MatrixXd> generate_data() {
        // Generate training data
        std::vector<MatrixXd> ans;
        
        // Generate training data
        VectorXd X_train_vec = VectorXd::LinSpaced(n, -1, 1);
        VectorXd Y_train_vec = func(X_train_vec) + sigma_y * (VectorXd::Random(n) * 0.5);
        // test data
        VectorXd X_test_vec = VectorXd::LinSpaced(1000, -1.5, 1.5);
        VectorXd f_true_vec = func(X_test_vec);
        // inducing points
        VectorXd X_m_vec = VectorXd::LinSpaced(m, -0.4, 0.4);

        // Convert to single column matrices using Map (no copy)
        MatrixXd X_train = Map<MatrixXd>(X_train_vec.data(), n, 1);
        MatrixXd Y_train = Map<MatrixXd>(Y_train_vec.data(), n, 1);
        MatrixXd X_test = Map<MatrixXd>(X_test_vec.data(), 1000, 1);
        MatrixXd f_true = Map<MatrixXd>(f_true_vec.data(), 1000, 1);
        MatrixXd X_m = Map<MatrixXd>(X_m_vec.data(), m, 1);

        ans.push_back(X_train);
        ans.push_back(Y_train);
        ans.push_back(X_test);
        ans.push_back(f_true);
        ans.push_back(X_m);

        return ans;
    }


    MatrixXd isotropic_rbf(const MatrixXd& X1, const MatrixXd&  X2, const VectorXd& theta){
        /*
        X1: matrix of m points (d, m)
        X2: matrix of n points (d, n)
        theta: kernel parameters: length scale and variance amplitude
        return: covariance matrix (m, n)
        */
        VectorXd X1_sqnorms = X1.colwise().squaredNorm();
        VectorXd X2_sqnorms = X2.colwise().squaredNorm();

        MatrixXd sqdist = X1_sqnorms.replicate(1, X2.cols()) + 
                         X2_sqnorms.transpose().replicate(X1.cols(), 1) - 
                         2.0 * X1.transpose() * X2;

        // Apply RBF kernel
        double length_scale_sq = theta[0] * theta[0];
        double variance = theta[1] * theta[1];
        return variance * (-0.5 / length_scale_sq * sqdist.array()).exp().matrix();
    }

    VectorXd kernel_diag(int d, const VectorXd& theta){
        // Creates a vector of length d filled with theta[1]^2 (variance)
        return VectorXd::Constant(d, theta[1] * theta[1]);
    }

    MatrixXd jitter(int d, double value = 1e-6){
        return MatrixXd::Identity(d, d) * value;
    }

    // verify accuracy
    // .array() converts a vector object to an array expression for element wise operations
    VectorXd softplus(const VectorXd& x) {
        return (1 + x.array().exp()).log();
    }
    
    VectorXd softplus_inv(const VectorXd& x) {
        return (x.array().exp() - 1.0).log();
    }

    VectorXd pack_params(const VectorXd& theta, const MatrixXd& X_m) {
        VectorXd theta_transformed = softplus_inv(theta);
        // create a Map object that interprets the raw memory of X_m as a 1D vector of doubles with as many elements; const ensures the map data isn't modified
        VectorXd X_m_flat = Map<const VectorXd>(X_m.data(), X_m.size());
        
        VectorXd packed(theta_transformed.size() + X_m_flat.size());
        packed << theta_transformed, X_m_flat;
        
        return packed;
    }
    
    std::pair<VectorXd, MatrixXd> unpack_params(const VectorXd& params) {
        // Extract first 2 elements and apply softplus
        VectorXd theta = softplus(params.head(2));
        
        // Extract remaining elements and reshape to (-1, 1) matrix
        VectorXd remaining = params.tail(params.size() - 2);
        MatrixXd X_m = Map<MatrixXd>(remaining.data(), remaining.size(), 1);
        
        return std::make_pair(theta, X_m);
    }

    // Not done
    double nlb_fn(MatrixXd& X, VectorXd& y){
        nlb_rg
        return 0.0;
    }
    

    void minimize(){
        // implement LBGFS
        return;
    }

private:
    double nlb(const VectorXd& packed, const MatrixXd& X, const VectorXd& y) {
        /*
        Negative lower bound on log marginal likelihood
        packed: kernel parameters theta and inducing inputs X_m
        */
        
        // unpack the params
        std::pair<VectorXd, MatrixXd> params;
        params = unpack_params(packed);
        VectorXd theta = params.first;
        MatrixXd X_m = params.second;

        // compute covariance kernels
        MatrixXd K_mm = isotropic_rbf(X_m, X_m, theta);
        MatrixXd K_mn = isotropic_rbf(X_m, X, theta);

        // L - lower triangular factor from cholesky decomposition of K_mm
        // LLT<MatrixXd> llt(K_mm);
        // MatrixXd L = llt.matrixL();
        // assert(L.rows() == X_m.rows() && L.cols() == X_m.cols() && "L and X must be same shape");
        // MatrixXd A = L.triangularView<Lower>().solve(K_mn) / sigma_y;  // m x n
        // Cholesky decomposition
        LLT<MatrixXd> chol_mm(K_mm);
        if (chol_mm.info() != Success) {
            return std::numeric_limits<double>::infinity();
        }
        
        MatrixXd L_mm = chol_mm.matrixL();        // Lower triangular L
        
        
        // Use the solver directly instead of triangularView
        MatrixXd A = K_mm.colPivHouseholderQr().solve(K_mn) / sigma_y;  // m x n
        MatrixXd AAT = A * (A.transpose());          // m x m
        MatrixXd B = MatrixXd::Identity(AAT.rows(), AAT.cols()) + AAT; // m x m
        // Cholesky of B
        LLT<MatrixXd> chol_B(B);
        if (chol_B.info() != Success) {
            return std::numeric_limits<double>::infinity();
        } 
        
        // lower triangular of B
        MatrixXd L_B = chol_B.matrixL();        
        
        // 1/sigm_y * ( L_B \ (A*y) )
        VectorXd c = L_B.colPivHouseholderQr().solve(A * y) / sigma_y;

        // solving for lower bound lb
        double lb = - n/2.0 * log(2*M_PI); 
        lb -= L_B.diagonal().array().log().sum(); // log|B|
        lb -= n / 2.0 * log(sigma_y*sigma_y);
        lb -= 0.5 / (sigma_y*sigma_y) * (y.dot(y));
        lb += 0.5 * c.dot(c);
        lb -= 0.5 / (sigma_y*sigma_y) * kernel_diag(n, theta).sum();
        lb += 0.5 * AAT.trace();
        return lb;

    }

    // write a unit test

};


class Tools {
public: 
    Tools() {}
    ~Tools() {}

    void writeToCSV(const MatrixXd& X, const MatrixXd& Y, const std::string& filename) {
        assert(X.rows() == Y.rows() && X.cols() == Y.cols() && "X and Y must be same shape");
        std::ofstream file(filename);
        file << "X,Y\n";
        for (size_t i = 0; i < X.rows(); ++i) {
            for (size_t j = 0; j < X.cols(); ++j){
                file << X(i, j) << "," << Y(i, j) << "\n";
            }
        }
        file.close();
        std::cout << "Data written to " << filename << std::endl;
    }

};


int main(){
    // Number of training examples
    int n = 100;
    // Number of inducing variables
    int m = 30; 
    // Noise
    double sigma_y = 0.2;
    SparseGP1D gp(n,m,sigma_y);
    std::vector< MatrixXd > ans = gp.generate_data();

    Tools tool;
    // Visualize training data against the true function
    tool.writeToCSV(ans[0], ans[1], "trainingdata.csv");
    tool.writeToCSV(ans[2], ans[3], "latentfunction.csv");
    return 0;
}