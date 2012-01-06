#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>

#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "util.hpp"
#include "data_loader.hpp"


using namespace std;
using namespace boost::numeric::ublas;


bool debug = true;

//
double sigmoid(double x) {
    double e = 2.718281828;

    return 1.0 / (1.0 + pow(e, -x));
}


// target: max { sum {log f(y(i)z(i)}} for i in (1, n) where f(x) = 1/1+e**(-x)
// and z(i) = sum(w(k) * x(i)(k)) for k in (1, l) where i denotes the ith training instance
// and k denotes the kth feature. 
// The gradient of the log-likehood with respect to the kth weight is:
// gra = sum{y(i)x(i)(k)f(-y(i)z(i))}, then we know how to update the weight in each iteration:
// w(k)(t+1) = w(k)(t) + e * gra
void lr_without_regularization(boost::numeric::ublas::matrix<double>& x,
        boost::numeric::ublas::vector<double>& y
        ) {

    // the convergence rate
    double epsilon = 0.0001;
    // the learning rate
    double gamma = 0.00005;
    int max_iters = 2000;
    int iter = 0;

    // init
    boost::numeric::ublas::vector<double> weight_old(x.size2());
    for (size_t i=0; i<weight_old.size(); ++i) {
        weight_old(i) = 0; 
    } 

    cout << "old weight: " << weight_old << endl;

    boost::numeric::ublas::vector<double> weight_new(x.size2());
    for (size_t i=0; i<weight_new.size(); ++i) {
        weight_new(i) = 0; 
    } 
    cout << "new weight: " << weight_new << endl;
    
    while (true) {
        // update each weight
        for (size_t k=0; k<weight_new.size(); ++k) {
            double gradient = 0;
            for (size_t i=0; i<x.size1(); ++i) {
                double z_i = 0;
                for (size_t j=0; j<weight_old.size(); ++j) {
#if 0
                    cout << "x(i,j):" << x(i,j) << endl;
                    cout << "weight_old(j):" << weight_old(j) << endl;
#endif
                    z_i += weight_old(j) * x(i,j);
                }
#if 0
                cout << "z_i:" << z_i << endl;
                cout << "y(i):" << y(i) << endl;
                cout << "x(i,k)" << x(i, k) << endl;
                cout << "sigmoid(-y(i) * z_i)" << sigmoid(-y(i) * z_i) << endl; 
#endif
                gradient = y(i) * x(i,k) * sigmoid(-y(i) * z_i);
            }
            weight_new(k) = weight_old(k) + gamma * gradient;
        }

        double dist = norm(weight_new, weight_old);
        if (dist < epsilon) {
            cout << "the best weight: " << weight_new << endl;
            break;
        }
        else {
            weight_old.swap(weight_new);
           // weight_old = weight_new;
        }

        iter += 1;
        if (iter >= max_iters) {
            cout << "Reach max_iters=" << max_iters << endl;
            break;
        }
        
        cout << "================================================" << endl;
        cout << "The " << iter << " th iteration, weight:" << endl;
        cout << weight_new << endl << endl;
        cout << "the diff between the old weight and the new weight: " << dist << endl << endl;
    }

    cout << "The best weight:" << endl;
    cout << weight_new << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " data_file" << endl;
        return -1;
    }

    const int record_num = 270;
    const int dim_num = 13 + 1;

    boost::numeric::ublas::vector<double> y(record_num);
    boost::numeric::ublas::matrix<double> x(record_num, dim_num);
    SimpleDataLoader loader(record_num, dim_num);
    loader.load_file(argv[1], y, x);

    // lr_method
    lr_without_regularization(x, y);

    return 0;
}
