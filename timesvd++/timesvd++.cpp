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


// The implementation here is based off the Bellkor solution paper, and the 
// timeSVD++ model. 



/* Splits each line of all.dta by spaces, then converts it to a double, and
 * finally returns a vector of doubles */ 
inline std::vector<double> split_by_whitespace(string s) {
    vector<double> result;
    istringstream iss(s);
    for(string s; iss >> s; )
        result.push_back(std::stod(s));
    return result;
}

/* Loads all the data into training data. 
 * ____data[i][0] = user id of i^th rating
 * ____data[i][1] = movie id of i^th rating
 * ____data[i][2] = date of i^th rating. See README in um file for details.
 * ____data[i][3] = rating. 1 - 5 is actual rating and 0 means blank. 
 * Now, we have blanks above because there's a bunch of different types of 
 * data. Specifically, we have training_data, validation_data, hidden_data, 
 * probe_data, and qual_data. We will store all of these in a vector, and
 * then return the vector. 
 */
inline vector<vector<vector<double> > > get_training_data(int PERCENT) {
    fstream afile; string str; int counter = 0; vector<double> temp;
    fstream bfile; string identifier; double id; 
    vector<vector<double> > training_data;
    vector<vector<double> > validation_data;
    vector<vector<double> > hidden_data;
    vector<vector<double> > probe_data;
    vector<vector<double> > qual_data;
    afile.open("../../mu/all.dta", ios::out | ios::in); 
    bfile.open("../../mu/all.idx", ios::out | ios::in); 
    if (PERCENT == 100) {
        int one_percent = ALL_DATA_SIZE/100;
        while ((std::getline(afile, str)) && \
            (std::getline(bfile, identifier)))  {
            temp = split_by_whitespace(str);
            id = stod(identifier);
            if (temp[3] == 0) {
                assert (id == 5);
            }
            if (id <= 3.0) {
                training_data.push_back(temp);
            }
            /*
            else if (id == 2.0) {
                validation_data.push_back(temp);
            }
            else if (id == 3.0) {
                hidden_data.push_back(temp);
            }*/
            else if (id == 4.0) {
                probe_data.push_back(temp);
            }
            else if (id == 5.0) {
                qual_data.push_back(temp);
            }
            else {
                assert (false);
            }
            assert (temp.size() == 4);
            counter += 1;
            if (counter % one_percent == 0) {
                // cout << counter << endl;
                cout.flush();
                cout << "\r" << "LOADING IN DATA:  " << counter/one_percent \
                    << " PERCENT COMPLETED" ;
            }
        }
    }
    else {
        int stop_at = int(double(ALL_DATA_SIZE * PERCENT)/100);
        int one_percent = stop_at/100;
        while (((std::getline(afile, str)) && \
            (std::getline(bfile, identifier))) && (counter < stop_at))  {
            temp = split_by_whitespace(str);
            id = stod(identifier);
            if (temp[3] == 0) {
                assert (id == 5);
            }
            if (id <= 3.0) {
                training_data.push_back(temp);
            }/*
            else if (id == 2.0) {
                validation_data.push_back(temp);
            }
            else if (id == 3.0) {
                hidden_data.push_back(temp);
            }*/
            else if (id == 4.0) {
                probe_data.push_back(temp);
            }
            else if (id == 5.0) {
                qual_data.push_back(temp);
            }
            else {
                assert (false);
            }
            assert (temp.size() == 4);
            counter += 1;
            if (counter % one_percent == 0) {
                // cout << counter << endl;
                cout.flush();
                cout << "\r" << "LOADING IN DATA:  " << counter/one_percent \
                    << " PERCENT COMPLETED" ;
            }
        }
    }
    printf("\n");
    afile.close();
    bfile.close();
    vector<vector<vector<double> > > all_data;
    all_data.push_back(training_data);
    all_data.push_back(validation_data);
    all_data.push_back(hidden_data);
    all_data.push_back(probe_data);
    all_data.push_back(qual_data);
    return all_data; 
}

// Calculates time deviation.
inline double calc_dev_u(double t, double Beta, double user_avg_time) {
    double sub = t - user_avg_time;
    return (sub > 0) ? pow(sub, Beta) : (-1 * pow(-1 * sub, Beta));
}


// Calculates bin_number_of time
inline int calc_bin(double t, int num_bins, int max_days) {
    return int(t/(max_days/num_bins));
} 

// The user and movie number for this function WONT be the user number, and 
// movie number, but rather the indexes, so MAKE sure you subtract 1 from this.
inline double calc_bui(int user, int movie, double t, double average, 
    double* user_biases, double* alpha, double* user_rating_time, double Beta, 
    map<pair<double, double>, double> &b_ut, double* item_bias, 
    double** item_bin, map<pair<double, double>, double> &c_u_t, double* 
    cu) {
    // Verify that we're 1 indexed. 
    if (user == 458293 || movie == 17770) {
        assert (false);
    }
    double dev_u_tui = calc_dev_u(t, Beta, user_rating_time[user]);
    int bin_no = calc_bin(t, NUM_BINS, MAX_DAYS);
    double b_i_bin_tui = item_bin[movie][bin_no];
    double c_u_tui = c_u_t[make_pair(double(user), double(t))] + cu[user];
    double b_u_tui = b_ut[make_pair(double(user), double(t))];

    double b_ui = average + user_biases[user] + alpha[user] * dev_u_tui + \
        b_u_tui + (item_bias[movie] + b_i_bin_tui) * c_u_tui;
    return b_ui;
}

inline double predict_rating(double avg, double* item_bias, double* user_bias, 
    double** U, double** V) {
    return 1.0; 
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

inline void initialize_avg_time(double** TRAINING_DATA, \
    double* USER_AVG_TIME, double* user_counting, int training_size) {
    double* temp = 0;
    int u = 0;
    double t_uj = 0.0;
    for (int i = 0; i < training_size; i++) {
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
    initialize_zeros_1d(user_counting, NUM_USERS);

}
inline void initialize_b_ut_map(map<pair<double, double>, double> &m, \
    double** TRAINING_DATA, int size) {
    double u = 0.0;
    for (int i = 0; i < size; i++) {
        double* temp = TRAINING_DATA[i];
        u = temp[0] - 1;
        double t_uj = temp[2];
        m[make_pair(u, t_uj)] = 0.0;
    }
}

inline void initialize_c_ut_map(map<pair<double, double>, double> &m, \
    double** TRAINING_DATA, int size) {
    double u = 0.0;
    for (int i = 0; i < size; i++) {
        double* temp = TRAINING_DATA[i];
        u = temp[0] - 1;
        double t_uj = temp[2];
        m[make_pair(u, t_uj)] = 1.0;
    }
}

int main(int argc, char* argv[]) {
    // THERE'S GONNA BE A SHIT TON OF PREPROCESSING. LIKE A LOT. HOLD ON IT'S 
    // GONNA BE A WILD RIDE. 
    clock_t begin = clock();
    if (argc != 2) {
        cout << "usage: ./FILENAME   (N Percent Of Data)" << endl;
        return 0;
    }

    // Allocate all appropriate memory. 
    int PERCENT = stoi((argv[1]));
    double* USER_BIASES = new double [NUM_USERS];
    double* ALPHAS_USERS = new double [NUM_USERS];
    double* user_counting = new double [NUM_USERS];
    double* movie_counting = new double [NUM_MOVIES];
    double* C_u = new double [NUM_USERS];
    double* USER_AVG_TIME = new double [NUM_USERS];
    double* MOVIE_BIASES = new double [NUM_MOVIES];
    double** B_I_BIN = new double* [NUM_MOVIES];
    map<pair<double, double>, double> b_u_tui;
    map<pair<double, double>, double> c_u_tui;
    for (int i = 0; i < NUM_MOVIES; i++) {
        B_I_BIN[i] = new double[NUM_BINS];
    }

    // Initialize arrays to 0. 
    initialize_rand_1d(USER_BIASES, NUM_USERS);
    initialize_rand_1d(ALPHAS_USERS, NUM_USERS);
    initialize_zeros_1d(USER_AVG_TIME, NUM_USERS);
    initialize_rand_1d(MOVIE_BIASES, NUM_MOVIES);
    initialize_rand_2d(B_I_BIN, NUM_MOVIES, NUM_BINS);
    initialize_ones_1d(C_u, NUM_USERS);

    // So far everything is negligble time. 
    vector<vector<vector<double> > > full_data = get_training_data(PERCENT);
    vector<vector<double> > training_data = full_data[0];
    vector<vector<double> > validation_data = full_data[3];
    vector<vector<double> > qual_data = full_data[4];

    int NUM_POINTS = training_data.size();
    int POINT_WIDTH = training_data[0].size();
    int NUM_VALIDATION = validation_data.size();
    cout << "VALIDATON SET SIZE:   " << NUM_VALIDATION << endl;
    cout << "TEST SET SIZE:   " << qual_data.size() << endl;
    if (PERCENT == 100) {
        assert (qual_data.size() == 2749898);
    }
    // Setting up training data. 
    double** TRAINING_DATA = new double* [NUM_POINTS];
    for (int i = 0; i < NUM_POINTS; i++) {
        TRAINING_DATA[i] = new double[POINT_WIDTH];
    }
    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < POINT_WIDTH; j++) {
            TRAINING_DATA[i][j] = training_data[i][j]; 
        }
    }

    // Setting up validation data. 
    double** VALIDATION_DATA = new double* [NUM_VALIDATION];
    for (int i = 0; i < NUM_VALIDATION; i++) {
        VALIDATION_DATA[i] = new double[POINT_WIDTH];
    }
    for (int i = 0; i < NUM_VALIDATION; i++) {
        for (int j = 0; j < POINT_WIDTH; j++) {
            VALIDATION_DATA[i][j] = validation_data[i][j]; 
        }
    }
    double average_rating = 0.0;
    for (int i = 0; i < NUM_POINTS; i++) {
        average_rating += TRAINING_DATA[i][3];
    }
    average_rating /= NUM_POINTS;
    cout << "average_rating: " << average_rating << endl;

    cout << "Initializing Average Rating Time User Vector:  " << endl;
    initialize_avg_time(TRAINING_DATA, USER_AVG_TIME, user_counting, NUM_POINTS);
    cout << "Initializing User Day Specific Bias Map:  " << endl;
    initialize_b_ut_map(b_u_tui, TRAINING_DATA, NUM_POINTS);
    cout << "Initializing User Specific Scaling Factor:   " << endl;
    initialize_c_ut_map(c_u_tui, TRAINING_DATA, NUM_POINTS);
    clock_t end = clock();
    cout << "TOTAL INITIALIZATION TIME:  " << \
        double(end - begin)/CLOCKS_PER_SEC << endl; 


    int M = NUM_USERS;
    int N = NUM_POINTS;
    int K = 20;
    int max_epochs = 250;
    double eps = 0.00000000000001;

    // Set up U matrix. 
    double** U = new double* [NUM_USERS];
    for (int i = 0; i < NUM_USERS; i++) {
        U[i] = new double[K];
    }

    // Set up V matrix. 
    double** V = new double* [NUM_MOVIES];
    for (int i = 0; i < NUM_MOVIES; i++) {
        V[i] = new double[K];
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



}