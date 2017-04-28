#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <sstream> 
#include <assert.h> 
#include <functional>
#include <numeric>
#include <math.h>
#include <stdlib.h>  
#include "../submission.cpp"
#include "loading_in_functions.cpp"
#include <utility> 
#include <map>
#include <ctime>
#define NUM_USERS 458293
#define NUM_MOVIES 17770
#define ALL_DATA_SIZE 102416306
#define MODEL_TYPE "TIME_SVD++"
#define NUM_BINS 30
#define MAX_DAYS 100
using namespace std;



// GLOBAL VARIABLE DECLARATION. REDUCES A LOT OF FUNCTION OVERHEAD. MAKES 
// EVERYTHING READABLE. 
double* USER_BIASES; // 
double* ALPHAS_USERS;
double* user_counting;
double* movie_counting;
double* C_u;
double* USER_AVG_TIME;
double* MOVIE_BIASES;
double** B_I_BIN;
double** U;
double** V; 
double** Y;
double** ALPHAS_UK;
double Beta = 0.4;
vector<vector<double> > MOVIES_RATED_BY_USERS; 
map<pair<double, double>, double> b_u_tui;
map<pair<double, double>, double> c_u_tui;
int K = 20;
double** TRAINING_DATA;
double** VALIDATION_DATA;
double average_rating = 0.0;
int NUM_POINTS = 0;
int NUM_VALIDATION = 0;
int POINT_WIDTH = 4;

// All learning parameters.

// b_ui paramaters.  
double l_rate_bu = 0.003;
double reg_bu = 0.03;
double l_rate_b_ut = 0.00025;
double reg_b_ut = 0.005;
double l_rate_alpha_u = 0.0001;
double reg_alpha_u = 50;
double l_rate_bi = 0.002;
double reg_bi = 0.03;
double l_rate_b_i_bin = 0.0005;
double reg_b_i_bin = 0.1;
double l_rate_c_u = 0.008;
double reg_c_u = 0.01;
double l_rate_c_ut = 0.002;
double reg_c_ut = 0.005;

// vector_prod parameters. 
double l_rate_v_u_y = 0.008;
double reg_v_u_y = 0.008;
double l_rate_alphas_uk = 0.00001;
double reg_alphas_uk = 50;
    


// The implementation here is based off the Bellkor solution paper, and the 
// timeSVD++ model. 


// Dot Product
inline double dot_product(double* A, double* B, int size) {
    double tot = 0.0;
    for (int i = 0; i < size; i++) 
        tot += A[i] * B[i];
    return tot;
}


// Calculates time deviation.
inline double calc_dev_u(double t, double user_avg_time) {
    double sub = t - user_avg_time;
    return (sub > 0) ? pow(sub, Beta) : (-1 * pow(-1 * sub, Beta));
}


// Calculates bin_number_of time
inline int calc_bin(double t) {
    return int(t/(MAX_DAYS/NUM_BINS));
} 

// The user and movie number for this function WONT be the user number, and 
// movie number, but rather the indexes, so MAKE sure you subtract 1 from this.
inline double calc_bui(int user, int movie, double t) {
    // Verify that we're 1 indexed. 
    if (user == 458293 || movie == 17770) {
        assert (false);
    }
    double dev_u_tui = calc_dev_u(t, USER_AVG_TIME[user]);
    int bin_no = calc_bin(t);
    double b_i_bin_tui = B_I_BIN[movie][bin_no];
    double c_u_part_2 = C_u[user];
    double c_u_part_1 = 1.0;
    double pot = c_u_tui[make_pair(double(user), double(t))];
    if (pot != 0) {
        c_u_part_1 = pot;
    }
    double c_u_scale = c_u_part_1 + c_u_part_2;
    double b_u_day_bias = b_u_tui[make_pair(double(user), double(t))];

    double b_ui = average_rating + USER_BIASES[user] + 
        ALPHAS_USERS[user] * dev_u_tui + b_u_day_bias + (MOVIE_BIASES[movie] + 
        b_i_bin_tui) * c_u_scale;
    return b_ui;
}

inline double calc_inner_vector_product(int user, int movie, double t) {
    double tot = 0.0;
    double* temp = V[movie];
    for (int i = 0; i < K; i++) {
        double second_v = U[user][i] + ALPHAS_UK[user][i] * 
            calc_dev_u(t, USER_AVG_TIME[user]);
        int num_movies_rated = MOVIES_RATED_BY_USERS[user].size();
        double second_tot = 0.0;
        for (int j = 0; j < num_movies_rated; j++) {
            second_tot += Y[int(MOVIES_RATED_BY_USERS[user][j])][i];
        }
        assert (!(isnan(second_v)));
        if (num_movies_rated == 0) {
            second_tot *= 0;
        }
        else {
            second_tot *= pow(num_movies_rated, -0.5);
        }
        assert (!(isnan(second_tot)));
        assert(!(isnan(temp[i])));
        tot += temp[i] * (second_tot + second_v);
    }
    return tot;

}

inline double predict_rating(int user, int movie, double t) {
    double prod = calc_inner_vector_product(user, movie, t);
    double bui = calc_bui(user, movie, t);
    assert (!(isnan(prod)));
    assert (!(isnan(bui)));
    if (prod + bui < 1) {
        return 1;
    }
    else if (prod + bui > 5) {
        return 5;
    } 
    return prod + bui; 
}

inline double get_err_val() {
    double err = 0.0;
    double r = 0.0;
    double p_r = 0.0;
    for (int i = 0; i < NUM_VALIDATION; i++) {
        double* temp = VALIDATION_DATA[i];
        int u = int(temp[0] - 1);
        int i_m = int(temp[1] - 1);
        int t = int(temp[2] - 1);
        r = int(temp[3] - 1);
        p_r = predict_rating(u, i_m, t); 
        err += pow(p_r - r, 2);
    }
    return pow(err/double(NUM_VALIDATION), 0.5);
}


inline void initialize_zeros_1d(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = 0.0;
    }
}

inline void initialize_ones_1d(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = 1.0;
    }
}

inline void initialize_rand_1d(double* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = double((rand() % 10000))/(10000) - 0.5;
    }
}

inline void initialize_rand_2d(double** arr, int size_d1, int size_d2) {
    for (int i = 0; i < size_d1; i++) {
        for (int j = 0; j < size_d2; j++) {
            arr[i][j] = double((rand() % 10000))/(10000) - 0.5;
        }
    }
}

inline void initialize_avg_time() {
    double* temp = 0;
    int u = 0;
    double t_uj = 0.0;
    initialize_zeros_1d(user_counting, NUM_USERS);
    for (int i = 0; i < NUM_POINTS; i++) {
        temp = TRAINING_DATA[i];
        u = int(temp[0] - 1);
        t_uj = temp[2];
        USER_AVG_TIME[u] = USER_AVG_TIME[u] + t_uj;
        user_counting[u] += 1;
    }
    for (int i = 0; i < NUM_USERS; i++) {
        if (user_counting[i] > 0) {
            USER_AVG_TIME[i] = double(USER_AVG_TIME[i])/(user_counting[i]);
        }
        else {
            USER_AVG_TIME[i] = 0;
        }
    }

}
inline void initialize_maps() {
    double u = 0.0;
    for (int i = 0; i < NUM_POINTS; i++) {
        double* temp = TRAINING_DATA[i];
        u = temp[0] - 1;
        double t_uj = temp[2];
        b_u_tui[make_pair(u, t_uj)] = 0.0;
        c_u_tui[make_pair(u, t_uj)] = 1.0;
    }
}


inline void gradient_updates(int user, int movie, double t, double r) {
    double e = r - predict_rating(user, movie, t);
    USER_BIASES[user] = USER_BIASES[user] + 
        l_rate_bu * (e - reg_bu * USER_BIASES[u]);
    MOVIE_BIASES[movie] = MOVIE_BIASES[movie] + 
        l_rate_bi * (e - reg_bi * MOVIE_BIASES[movie]);
    double* user_counting;
    double* movie_counting;
    double* C_u;
    double* USER_AVG_TIME;
    double* MOVIE_BIASES;
    double** B_I_BIN;
    double** U;
    double** V; 
    double** Y;
    double** ALPHAS_UK;
    double Beta = 0.4;
    vector<vector<double> > MOVIES_RATED_BY_USERS; 
    map<pair<double, double>, double> b_u_tui;
    map<pair<double, double>, double> c_u_tui;



}


int main(int argc, char* argv[]) {
    // THERE'S GONNA BE A SHIT TON OF PREPROCESSING. LIKE A LOT. HOLD ON IT'S 
    // GONNA BE A WILD RIDE. 
    clock_t begin = clock();
    if (argc != 2) {
        cout << "usage: ./FILENAME   (N Percent Of Data)" << endl;
        return 0;
    }
    int PERCENT = stoi((argv[1]));

    cout << PERCENT << endl;

    // Allocate all appropriate memory. 
    USER_BIASES = new double [NUM_USERS];
    ALPHAS_USERS = new double [NUM_USERS];
    user_counting = new double [NUM_USERS];
    movie_counting = new double [NUM_MOVIES];
    C_u = new double [NUM_USERS];
    USER_AVG_TIME = new double [NUM_USERS];
    MOVIE_BIASES = new double [NUM_MOVIES];
    B_I_BIN = new double* [NUM_MOVIES];
    Beta = 0.4;
    // vector<vector<double> > MOVIES_RATED_BY_USERS; 
    // map<pair<double, double>, double> b_u_tui;
    // map<pair<double, double>, double> c_u_tui;
    for (int i = 0; i < NUM_MOVIES; i++) {
        B_I_BIN[i] = new double[NUM_BINS];
    }

    // Initialize arrays to 0. 
    initialize_rand_1d(USER_BIASES, NUM_USERS);
    initialize_rand_1d(ALPHAS_USERS, NUM_USERS);
    initialize_zeros_1d(USER_AVG_TIME, NUM_USERS);
    initialize_zeros_1d(user_counting, NUM_USERS);
    initialize_rand_1d(MOVIE_BIASES, NUM_MOVIES);
    initialize_rand_2d(B_I_BIN, NUM_MOVIES, NUM_BINS);
    initialize_ones_1d(C_u, NUM_USERS);

    // So far everything is negligble time. 
    vector<vector<vector<double> > > full_data = get_training_data(PERCENT);
    vector<vector<double> > training_data = full_data[0];
    vector<vector<double> > validation_data = full_data[3];
    vector<vector<double> > qual_data = full_data[4];

    NUM_POINTS = training_data.size();
    POINT_WIDTH = training_data[0].size();
    NUM_VALIDATION = validation_data.size();
    cout << "VALIDATON SET SIZE:   " << NUM_VALIDATION << endl;
    cout << "TEST SET SIZE:   " << qual_data.size() << endl;
    if (PERCENT == 100) {
        assert (qual_data.size() == 2749898);
    }


    // Setting up MOVIES_RATED_BY_USERS 2D vector. 
    for (int i = 0; i < NUM_USERS; i++) {
        vector<double> temp; 
        MOVIES_RATED_BY_USERS.push_back(temp);
    }

    // Setting up training data. 
    TRAINING_DATA = new double* [NUM_POINTS];
    for (int i = 0; i < NUM_POINTS; i++) {
        TRAINING_DATA[i] = new double[POINT_WIDTH];
    }

    // Setting up movies rated by users as well. 
    for (int i = 0; i < NUM_POINTS; i++) {
        vector<double> temp = training_data[i];
        MOVIES_RATED_BY_USERS[int(temp[0] - 1)].push_back(int(temp[1] - 1));
        for (int j = 0; j < POINT_WIDTH; j++) {
            TRAINING_DATA[i][j] = training_data[i][j]; 
        }
    }
    // Setting up validation data. 
    VALIDATION_DATA = new double* [NUM_VALIDATION];
    for (int i = 0; i < NUM_VALIDATION; i++) {
        VALIDATION_DATA[i] = new double[POINT_WIDTH];
    }
    for (int i = 0; i < NUM_VALIDATION; i++) {
        for (int j = 0; j < POINT_WIDTH; j++) {
            VALIDATION_DATA[i][j] = validation_data[i][j]; 
        }
    }
    average_rating = 0.0;
    for (int i = 0; i < NUM_POINTS; i++) {
        average_rating += TRAINING_DATA[i][3];
    }
    average_rating /= NUM_POINTS;
    cout << "average_rating: " << average_rating << endl;

    cout << "Initializing Average Rating Time User Vector:  " << endl;
    initialize_avg_time();
    cout << "Initializing Day Specific Maps:  " << endl;
    initialize_maps();
    clock_t end = clock();
    cout << "TOTAL INITIALIZATION TIME:  "  
        << double(end - begin)/CLOCKS_PER_SEC << endl; 


    // int M = NUM_USERS;
    // int N = NUM_POINTS;
    // int K = 20;
    // int max_epochs = 2;
    // double eps = 0.00000000000001;

    // Set up U matrix. 
    U = new double* [NUM_USERS];
    for (int i = 0; i < NUM_USERS; i++) {
        U[i] = new double[K];
    }

    // Set up V matrix. 
    V = new double* [NUM_MOVIES];
    for (int i = 0; i < NUM_MOVIES; i++) {
        V[i] = new double[K];
    }

    // Set up Y matrix. 
    Y = new double* [NUM_MOVIES];
    for (int i = 0; i < NUM_MOVIES; i++) {
        Y[i] = new double[K];
    }

    // Set up Alpha_uk matrix. 
    ALPHAS_UK = new double* [NUM_USERS];
    for (int i = 0; i < NUM_USERS; i++) {
        ALPHAS_UK[i] = new double[K];
    }

    // Fill U in with random doubles. 
    for (int i = 0; i < NUM_USERS; i++) {
        for (int j = 0; j < K; j++ ) {
            U[i][j] = double((rand() % 1000) + 1)/1000 - 0.5;
        }
    }

    // Fill V in with random double. 
    for (int i = 0; i < NUM_MOVIES; i++) {
        for (int j = 0; j < K; j++ ) {
            V[i][j] = double((rand() % 1000) + 1)/1000 - 0.5;
        }
    }

    // Fill Y in with random doubles. 
    for (int i = 0; i < NUM_MOVIES; i++) {
        for (int j = 0; j < K; j++ ) {
            Y[i][j] = double((rand() % 1000) + 1)/1000 - 0.5;
        }
    }

    // Fill Alphas_uk in with random doubles. 
    for (int i = 0; i < NUM_USERS; i++) {
        for (int j = 0; j < K; j++) {
            ALPHAS_UK[i][j] = double((rand() % 1000) + 1)/1000 - 0.5;
        }
    }
    assert(!(isnan(V[1][19])));

    cout << "Validation Error: " << get_err_val() << endl;


    // Free everything. 
    free(C_u);
    free(USER_BIASES);
    free(ALPHAS_USERS);
    free(USER_AVG_TIME);
    free(MOVIE_BIASES);
    free(user_counting);
    free(movie_counting);
    for (int i = 0; i < NUM_MOVIES; i++) {
        free(B_I_BIN[i]);
    }
    free(B_I_BIN);
    for (int i = 0; i < NUM_POINTS; i++) {
        free(TRAINING_DATA[i]);
    }
    free(TRAINING_DATA);
    for (int i = 0; i < NUM_VALIDATION; i++) {
        free(VALIDATION_DATA[i]);
    }
    free(VALIDATION_DATA);
    for (int i = 0; i < NUM_USERS; i++) {
        free(U[i]);
    }
    free(U);
    for (int i = 0; i < NUM_MOVIES; i++) {
        free(V[i]);
    }
    free(V);
    for (int i = 0; i < NUM_MOVIES; i++) {
        free(Y[i]);
    }
    free(Y);
    for (int i = 0; i < NUM_MOVIES; i++) {
        free(ALPHAS_UK[i]);
    }
    free(ALPHAS_UK);

}