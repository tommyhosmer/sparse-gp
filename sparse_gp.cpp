#include <stdlib.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense"
#include "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/unsupported/Eigen/MatrixFunctions"
using namespace Eigen;


class SparseGP {
public:
    SparseGP() {}
    ~SparseGP() {}
    
    double func(double x){
        // latent function
        return sin(3*x*M_PI) + 0.3 * cos(x*9*M_PI) + 0.5 * sin(x*7*M_PI);
    }

    std::vector< std::vector<double> > generate_data(int n, int m, float sigma_y) {
        // Generate training data
        std::vector< std::vector<double> > ans;
        // Noisy training data
        std::vector<double> X_train(n);
        std::vector<double> Y_train(n);
        // Test data
        std::vector<double> X_test(1000);
        std::vector<double> f_true(X_test.size());
        // Inducing points
        std::vector<double> X_m(m);


        for (int i = 0; i < n; ++i) {
            X_train[i] = -1 + 2. * i / (n - 1); // Equivalent to np.linspace(-1, 1, n)
            Y_train[i] = func(X_train[i]) + sigma_y * ((double)rand() / RAND_MAX - 0.5);
        }

        for (int i = 0; i < X_test.size(); ++i){
            X_test[i] = -1.5 + 3. * i /(X_test.size()-1);
            f_true[i] = func(X_test[i]);
        }

        for (int i = 0; i< m; ++i){
            X_m[i] = -0.4 + 0.8 * i / (m-1);
        }
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
    MatrixXd softplus(const MatrixXd& X){
       return (MatrixXd::Identity(X.rows(), X.cols()) + X.exp()).log();
    }
    
    MatrixXd softplus_inv(const MatrixXd& X){
       return (X.exp() - MatrixXd::Identity(X.rows(), X.cols())).log();
    }
    
    VectorXd pack_params(const VectorXd& theta, const MatrixXd& X_m) {
        VectorXd theta_transformed = softplus_inv(theta);
        VectorXd X_m_flat = Map<const VectorXd>(X_m.data(), X_m.size());
        
        VectorXd packed(theta_transformed.size() + X_m_flat.size());
        packed << theta_transformed, X_m_flat;
        
        return packed;
    }
    
    void minimize(){
        // implement LBGFS
        return;
    }

};

class Tools {
public: 
    Tools() {}
    ~Tools() {}

    void writeToCSV(const std::vector<double>& X, const std::vector<double>& Y, const std::string& filename) {
        std::ofstream file(filename);
        file << "X,Y\n";
        for (size_t i = 0; i < X.size(); ++i) {
            file << X[i] << "," << Y[i] << "\n";
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
    float sigma_y = 0.2;
    SparseGP gp;
    std::vector< std::vector<double> > ans = gp.generate_data(n, m, sigma_y);

    Tools tool;

    tool.writeToCSV(ans[0], ans[1], "trainingdata.csv");
    tool.writeToCSV(ans[2], ans[3], "latentfunction.csv");
    return 0;
}