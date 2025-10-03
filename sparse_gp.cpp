#include <stdlib.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include "/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/Eigen/Dense"
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

    double isotropic_rbf(std::vector< std::vector<double> > X1, std::vector<std::vector<double> >  X2, std::vector<std::vector<double> >   theta){
        /*
        X1: array of m points (m,d)
        X2: array of n points (n,d)
        
        */

        return ;
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