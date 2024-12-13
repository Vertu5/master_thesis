#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <limits>
#include <string>
#include "RGrid.h"
#include "VectorField.h"
#include "HHD.h"
#include "MaskedArray.h"

// Function to print masked arrays
template <typename T>
void print_field(const MaskedArray<T>& field, const std::string& name, size_t nx, size_t ny) {
    std::cout << "\n" << name << ":" << std::endl;
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            size_t idx = y * nx + x;
            if (field.ismasked(idx)) {
                std::cout << "   ---   ";
            } else {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << field[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
}

// Test 1: Uniform Flow with Single Masked Point
void test1_uniform_flow() {
    std::cout << "\nTest 1: Flux Uniforme avec Point Masqué Unique\n";

    // Créer la grille
    size_t nx = 4;
    size_t ny = 4;
    float dx = 1.0f;
    float dy = 1.0f;
    RGrid rgrid(nx, ny, dx, dy);

    // Initialiser le champ vectoriel
    VectorField<float> vfield;
    vfield.dim = 2;
    vfield.sz = nx * ny;
    vfield.u = MaskedArray<float>({ny, nx});
    vfield.v = MaskedArray<float>({ny, nx});

    // Définir les composantes U et V selon le test
    std::vector<float> U_values = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  std::nan(""), 7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f,14.0f, 15.0f, 16.0f
    };
    std::vector<float> V_values = {
        16.0f, 15.0f, 14.0f, 13.0f,
        12.0f,  std::nan(""), 10.0f,  9.0f,
        8.0f,  7.0f,  6.0f,  5.0f,
        4.0f,  3.0f,  2.0f,  1.0f
    };

    // Assigner les valeurs et appliquer les masques
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            // U component
            if (std::isnan(U_values[idx])) {
                vfield.u.mask[idx] = true;
            } else {
                vfield.u[idx] = U_values[idx];
                vfield.u.mask[idx] = false;
            }
            // V component
            if (std::isnan(V_values[idx])) {
                vfield.v.mask[idx] = true;
            } else {
                vfield.v[idx] = V_values[idx];
                vfield.v.mask[idx] = false;
            }
        }
    }

    // Calculer la divergence et la rotation
    vfield.need_divcurl(rgrid);

    // Afficher les résultats
    print_field(vfield.u, "Composante U", nx, ny);
    print_field(vfield.v, "Composante V", nx, ny);
    print_field(vfield.div, "Divergence", nx, ny);
    print_field(vfield.curlw, "Rotation", nx, ny);
}

// Test 2: Vortex Flow with Masked Region
void test2_vortex_flow() {
    std::cout << "\nTest 2: Flux Vortex avec Région Masquée\n";

    // Créer la grille
    size_t nx = 5;
    size_t ny = 5;
    float dx = 1.0f;
    float dy = 1.0f;
    RGrid rgrid(nx, ny, dx, dy);

    // Initialiser le champ vectoriel
    VectorField<float> vfield;
    vfield.dim = 2;
    vfield.sz = nx * ny;
    vfield.u = MaskedArray<float>({ny, nx});
    vfield.v = MaskedArray<float>({ny, nx});

    // Définir les composantes U et V selon le test
    std::vector<float> U_values = {
        -4.0f, -3.0f, -2.0f, -3.0f, -4.0f,
        -3.0f,  std::nan(""),  std::nan(""), std::nan(""), -3.0f,
        -2.0f,  std::nan(""),  std::nan(""), std::nan(""), -2.0f,
        -3.0f,  std::nan(""),  std::nan(""), std::nan(""), -3.0f,
        -4.0f, -3.0f, -2.0f, -3.0f, -4.0f
    };
    std::vector<float> V_values = {
        4.0f, 3.0f, 2.0f, 3.0f, 4.0f,
        3.0f,  std::nan(""),  std::nan(""), std::nan(""), 3.0f,
        2.0f,  std::nan(""),  std::nan(""), std::nan(""), 2.0f,
        3.0f,  std::nan(""),  std::nan(""), std::nan(""), 3.0f,
        4.0f, 3.0f, 2.0f, 3.0f, 4.0f
    };

    // Assigner les valeurs et appliquer les masques
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            // U component
            if (std::isnan(U_values[idx])) {
                vfield.u.mask[idx] = true;
            } else {
                vfield.u[idx] = U_values[idx];
                vfield.u.mask[idx] = false;
            }
            // V component
            if (std::isnan(V_values[idx])) {
                vfield.v.mask[idx] = true;
            } else {
                vfield.v[idx] = V_values[idx];
                vfield.v.mask[idx] = false;
            }
        }
    }

    // Calculer la divergence et la rotation
    vfield.need_divcurl(rgrid);

    // Afficher les résultats
    print_field(vfield.u, "Composante U", nx, ny);
    print_field(vfield.v, "Composante V", nx, ny);
    print_field(vfield.div, "Divergence", nx, ny);
    print_field(vfield.curlw, "Rotation", nx, ny);
}

// Test 3: Source Flow with Masked Boundary
void test3_source_flow() {
    std::cout << "\nTest 3: Flux Source avec Frontière Masquée\n";

    // Créer la grille
    size_t nx = 6;
    size_t ny = 6;
    float dx = 1.0f;
    float dy = 1.0f;
    RGrid rgrid(nx, ny, dx, dy);

    // Initialiser le champ vectoriel
    VectorField<float> vfield;
    vfield.dim = 2;
    vfield.sz = nx * ny;
    vfield.u = MaskedArray<float>({ny, nx});
    vfield.v = MaskedArray<float>({ny, nx});

    // Définir les composantes U et V selon le test
    std::vector<float> U_values = {
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""),
        std::nan(""), 2.0f, 4.0f, 6.0f, 8.0f, std::nan(""),
        std::nan(""), 10.0f, 12.0f, 14.0f, 16.0f, std::nan(""),
        std::nan(""), 18.0f, 20.0f, 22.0f, 24.0f, std::nan(""),
        std::nan(""), 26.0f, 28.0f, 30.0f, 32.0f, std::nan(""),
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan("")
    };
    std::vector<float> V_values = {
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""),
        std::nan(""), 3.0f, 6.0f, 9.0f, 12.0f, std::nan(""),
        std::nan(""), 15.0f, 18.0f, 21.0f, 24.0f, std::nan(""),
        std::nan(""), 27.0f, 30.0f, 33.0f, 36.0f, std::nan(""),
        std::nan(""), 39.0f, 42.0f, 45.0f, 48.0f, std::nan(""),
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan(""), std::nan("")
    };

    // Assigner les valeurs et appliquer les masques
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            // U component
            if (std::isnan(U_values[idx])) {
                vfield.u.mask[idx] = true;
            } else {
                vfield.u[idx] = U_values[idx];
                vfield.u.mask[idx] = false;
            }
            // V component
            if (std::isnan(V_values[idx])) {
                vfield.v.mask[idx] = true;
            } else {
                vfield.v[idx] = V_values[idx];
                vfield.v.mask[idx] = false;
            }
        }
    }

    // Masquer les points de la frontière
    for (size_t x = 0; x < nx; ++x) {
        vfield.u.mask[x] = true; // Ligne supérieure
        vfield.v.mask[x] = true;
        size_t idx_bottom = (ny - 1) * nx + x;
        vfield.u.mask[idx_bottom] = true; // Ligne inférieure
        vfield.v.mask[idx_bottom] = true;
    }
    for (size_t y = 0; y < ny; ++y) {
        size_t idx_left = y * nx;
        size_t idx_right = y * nx + (nx - 1);
        vfield.u.mask[idx_left] = true; // Colonne de gauche
        vfield.v.mask[idx_left] = true;
        vfield.u.mask[idx_right] = true; // Colonne de droite
        vfield.v.mask[idx_right] = true;
    }

    // Calculer la divergence et la rotation
    vfield.need_divcurl(rgrid);

    // Afficher les résultats
    print_field(vfield.u, "Composante U", nx, ny);
    print_field(vfield.v, "Composante V", nx, ny);
    print_field(vfield.div, "Divergence", nx, ny);
    print_field(vfield.curlw, "Rotation", nx, ny);
}

// Test 4: Flow with Diagonal Masked Region
void test4_diagonal_mask() {
    std::cout << "\nTest 4: Flux avec Région Diagonale Masquée\n";

    // Créer la grille
    size_t nx = 5;
    size_t ny = 5;
    float dx = 1.0f;
    float dy = 1.0f;
    RGrid rgrid(nx, ny, dx, dy);

    // Initialiser le champ vectoriel
    VectorField<float> vfield;
    vfield.dim = 2;
    vfield.sz = nx * ny;
    vfield.u = MaskedArray<float>({ny, nx});
    vfield.v = MaskedArray<float>({ny, nx});

    // Définir les composantes U et V selon le test
    std::vector<float> U_values = {
        std::nan(""), 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, std::nan(""), 8.0f, 9.0f, 10.0f,
        11.0f, 12.0f, std::nan(""), 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, std::nan(""), 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f, std::nan("")
    };
    std::vector<float> V_values = {
        std::nan(""), 24.0f, 23.0f, 22.0f, 21.0f,
        20.0f, std::nan(""), 18.0f, 17.0f, 16.0f,
        15.0f, 14.0f, std::nan(""), 12.0f, 11.0f,
        10.0f, 9.0f, 8.0f, std::nan(""), 6.0f,
        5.0f, 4.0f, 3.0f, 2.0f, std::nan("")
    };

    // Assigner les valeurs et appliquer les masques
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            // U component
            if (std::isnan(U_values[idx])) {
                vfield.u.mask[idx] = true;
            } else {
                vfield.u[idx] = U_values[idx];
                vfield.u.mask[idx] = false;
            }
            // V component
            if (std::isnan(V_values[idx])) {
                vfield.v.mask[idx] = true;
            } else {
                vfield.v[idx] = V_values[idx];
                vfield.v.mask[idx] = false;
            }
        }
    }

    // Calculer la divergence et la rotation
    vfield.need_divcurl(rgrid);

    // Afficher les résultats
    print_field(vfield.u, "Composante U", nx, ny);
    print_field(vfield.v, "Composante V", nx, ny);
    print_field(vfield.div, "Divergence", nx, ny);
    print_field(vfield.curlw, "Rotation", nx, ny);
}

// Test 5: Flow with Checkerboard Masked Pattern
void test5_checkerboard_mask() {
    std::cout << "\nTest 5: Flux avec Motif Masqué en Damier\n";

    // Créer la grille
    size_t nx = 6;
    size_t ny = 6;
    float dx = 1.0f;
    float dy = 1.0f;
    RGrid rgrid(nx, ny, dx, dy);

    // Initialiser le champ vectoriel
    VectorField<float> vfield;
    vfield.dim = 2;
    vfield.sz = nx * ny;
    vfield.u = MaskedArray<float>({ny, nx});
    vfield.v = MaskedArray<float>({ny, nx});

    // Définir les composantes U et V selon le test
    std::vector<float> U_values = {
        std::nan(""), 2.0f, std::nan(""), 4.0f, std::nan(""), 6.0f,
        7.0f, std::nan(""), 9.0f, std::nan(""), 11.0f, std::nan(""),
        std::nan(""), 14.0f, std::nan(""), 16.0f, std::nan(""), 18.0f,
        19.0f, std::nan(""), 21.0f, std::nan(""), 23.0f, std::nan(""),
        std::nan(""), 26.0f, std::nan(""), 28.0f, std::nan(""), 30.0f,
        31.0f, std::nan(""), 33.0f, std::nan(""), 35.0f, std::nan("")
    };
    std::vector<float> V_values = {
        std::nan(""), 35.0f, std::nan(""), 33.0f, std::nan(""), 31.0f,
        30.0f, std::nan(""), 28.0f, std::nan(""), 26.0f, std::nan(""),
        std::nan(""), 23.0f, std::nan(""), 21.0f, std::nan(""), 19.0f,
        18.0f, std::nan(""), 16.0f, std::nan(""), 14.0f, std::nan(""),
        std::nan(""), 11.0f, std::nan(""), 9.0f, std::nan(""), 7.0f,
        6.0f, std::nan(""), 4.0f, std::nan(""), 2.0f, std::nan("")
    };

    // Assigner les valeurs et appliquer les masques
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            // U component
            if (std::isnan(U_values[idx])) {
                vfield.u.mask[idx] = true;
            } else {
                vfield.u[idx] = U_values[idx];
                vfield.u.mask[idx] = false;
            }
            // V component
            if (std::isnan(V_values[idx])) {
                vfield.v.mask[idx] = true;
            } else {
                vfield.v[idx] = V_values[idx];
                vfield.v.mask[idx] = false;
            }
        }
    }

    // Calculer la divergence et la rotation
    vfield.need_divcurl(rgrid);

    // Afficher les résultats
    print_field(vfield.u, "Composante U", nx, ny);
    print_field(vfield.v, "Composante V", nx, ny);
    print_field(vfield.div, "Divergence", nx, ny);
    print_field(vfield.curlw, "Rotation", nx, ny);
}

// Test 6: Rotating Flow with Spiral Masked Pattern
void test6_rotating_mask() {
    std::cout << "\nTest 6: Flux Rotatif avec Motif Masqué en Spirale\n";

    // Créer la grille
    size_t nx = 7;
    size_t ny = 7;
    float dx = 1.0f;
    float dy = 1.0f;
    RGrid rgrid(nx, ny, dx, dy);

    // Initialiser le champ vectoriel
    VectorField<float> vfield;
    vfield.dim = 2;
    vfield.sz = nx * ny;
    vfield.u = MaskedArray<float>({ny, nx});
    vfield.v = MaskedArray<float>({ny, nx});

    // Définir les composantes U et V selon le test
    std::vector<float> U_values = {
        std::nan(""), 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        std::nan(""), std::nan(""), 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
        std::nan(""), std::nan(""), std::nan(""), 15.0f, 16.0f, 17.0f, 18.0f,
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), 21.0f, 22.0f, 23.0f,
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), 27.0f, 28.0f, 29.0f,
        std::nan(""), std::nan(""), 32.0f, 33.0f, 34.0f, 35.0f, 36.0f,
        std::nan(""), 37.0f, 38.0f, 39.0f, 40.0f, 41.0f, 42.0f
    };
    std::vector<float> V_values = {
        std::nan(""), 41.0f, 40.0f, 39.0f, 38.0f, 37.0f, 36.0f,
        std::nan(""), std::nan(""), 33.0f, 32.0f, 31.0f, 30.0f, 29.0f,
        std::nan(""), std::nan(""), std::nan(""), 25.0f, 24.0f, 23.0f, 22.0f,
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), 18.0f, 17.0f, 16.0f,
        std::nan(""), std::nan(""), std::nan(""), std::nan(""), 14.0f, 13.0f, 12.0f,
        std::nan(""), std::nan(""), std::nan(""), 8.0f, 7.0f, 6.0f, 5.0f,
        std::nan(""), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    // Assigner les valeurs et appliquer les masques
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            // U component
            if (std::isnan(U_values[idx])) {
                vfield.u.mask[idx] = true;
            } else {
                vfield.u[idx] = U_values[idx];
                vfield.u.mask[idx] = false;
            }
            // V component
            if (std::isnan(V_values[idx])) {
                vfield.v.mask[idx] = true;
            } else {
                vfield.v[idx] = V_values[idx];
                vfield.v.mask[idx] = false;
            }
        }
    }

    // Appliquer le masque en spirale (masquer les colonnes de gauche des couches)
    size_t layers = 3;
    for (size_t layer = 0; layer < layers; ++layer) {
        for (size_t j = layer; j < ny - layer; ++j) {
            size_t idx = j * nx + layer;
            vfield.u.mask[idx] = true;
            vfield.v.mask[idx] = true;
        }
    }

    // Calculer la divergence et la rotation
    vfield.need_divcurl(rgrid);

    // Afficher les résultats
    print_field(vfield.u, "Composante U", nx, ny);
    print_field(vfield.v, "Composante V", nx, ny);
    print_field(vfield.div, "Divergence", nx, ny);
    print_field(vfield.curlw, "Rotation", nx, ny);
}

// Test 7: Flow with Random Masked Points
void test7_random_mask() {
    std::cout << "\nTest 7: Flux avec Points Masqués Aléatoires\n";

    // Créer la grille
    size_t nx = 8;
    size_t ny = 8;
    float dx = 1.0f;
    float dy = 1.0f;
    RGrid rgrid(nx, ny, dx, dy);

    // Initialiser le champ vectoriel
    VectorField<float> vfield;
    vfield.dim = 2;
    vfield.sz = nx * ny;
    vfield.u = MaskedArray<float>({ny, nx});
    vfield.v = MaskedArray<float>({ny, nx});

    // Définir les composantes U et V selon le test
    std::vector<float> U_values = {
        1.0f,  2.0f,  3.0f,  4.0f, std::nan(""), 6.0f,  7.0f,  8.0f,
        std::nan(""), 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, std::nan(""),
        17.0f, std::nan(""), 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, std::nan(""), 27.0f, std::nan(""), std::nan(""), 30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f, std::nan(""), std::nan(""), 38.0f, std::nan(""), 40.0f,
        41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, std::nan(""),
        49.0f, std::nan(""), std::nan(""), std::nan(""), 53.0f, 54.0f, 55.0f, 56.0f,
        57.0f, 58.0f, std::nan(""), std::nan(""), 61.0f, 62.0f, 63.0f, 64.0f
    };
    std::vector<float> V_values = {
        64.0f, 63.0f, 62.0f, 61.0f, std::nan(""), 59.0f, 58.0f, 57.0f,
        std::nan(""), 55.0f, 54.0f, 53.0f, 52.0f, 51.0f, 50.0f, std::nan(""),
        48.0f, std::nan(""), 46.0f, 45.0f, 44.0f, 43.0f, 42.0f, 41.0f,
        40.0f, std::nan(""), 38.0f, std::nan(""), std::nan(""), 35.0f, 34.0f, 33.0f,
        32.0f, 31.0f, 30.0f, std::nan(""), std::nan(""), 27.0f, std::nan(""), 25.0f,
        24.0f, 23.0f, 22.0f, 21.0f, 20.0f, 19.0f, 18.0f, std::nan(""),
        16.0f, std::nan(""), std::nan(""), std::nan(""), 12.0f, 11.0f, 10.0f, 9.0f,
        8.0f, 7.0f, std::nan(""), std::nan(""), 4.0f, 3.0f, 2.0f, 1.0f
    };

    // Assigner les valeurs et appliquer les masques
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            size_t idx = i * nx + j;
            // U component
            if (std::isnan(U_values[idx])) {
                vfield.u.mask[idx] = true;
            } else {
                vfield.u[idx] = U_values[idx];
                vfield.u.mask[idx] = false;
            }
            // V component
            if (std::isnan(V_values[idx])) {
                vfield.v.mask[idx] = true;
            } else {
                vfield.v[idx] = V_values[idx];
                vfield.v.mask[idx] = false;
            }
        }
    }

    // Appliquer un masque aléatoire avec une probabilité de 30%
    std::srand(42);  // Graine fixe pour la reproductibilité
    for (size_t idx = 0; idx < vfield.sz; ++idx) {
        float rand_val = static_cast<float>(std::rand()) / RAND_MAX;
        if (rand_val < 0.3f) {
            vfield.u.mask[idx] = true;
            vfield.v.mask[idx] = true;
        }
    }

    // Calculer la divergence et la rotation
    vfield.need_divcurl(rgrid);

    // Afficher les résultats
    print_field(vfield.u, "Composante U", nx, ny);
    print_field(vfield.v, "Composante V", nx, ny);
    print_field(vfield.div, "Divergence", nx, ny);
    print_field(vfield.curlw, "Rotation", nx, ny);
}

int main() {
    // Exécuter les tests
    test1_uniform_flow();
    test2_vortex_flow();
    test3_source_flow();
    test4_diagonal_mask();
    test5_checkerboard_mask();
    test6_rotating_mask();
    test7_random_mask();

    return 0;
}
