#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <Eigen/Dense>
#include <vector>
#include <algorithm>

namespace interpolation {

/**
 * Interpolation linéaire 1D
 * @param x Points x d'entrée
 * @param y Valeurs y d'entrée
 * @param xi Points x à interpoler
 * @return Valeurs interpolées
 */
inline Eigen::VectorXd interp1_linear(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& xi) 
{
    int n = x.size();
    int ni = xi.size();
    Eigen::VectorXd yi(ni);

    for(int i = 0; i < ni; i++) {
        // Trouver l'intervalle contenant xi[i]
        auto it = std::lower_bound(x.data(), x.data() + n, xi[i]);
        int idx = std::distance(x.data(), it);
        
        if(idx == 0) {
            yi[i] = y[0];
        } else if(idx == n) {
            yi[i] = y[n-1];
        } else {
            // Interpolation linéaire
            double x0 = x[idx-1];
            double x1 = x[idx];
            double y0 = y[idx-1];
            double y1 = y[idx];
            yi[i] = y0 + (y1 - y0) * (xi[i] - x0) / (x1 - x0);
        }
    }
    
    return yi;
}

} // namespace interpolation

#endif // INTERPOLATION_HPP
