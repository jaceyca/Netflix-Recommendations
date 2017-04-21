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
using namespace std;
#define NUM_USERS 458293
#define NUM_MOVIES 17770
#define ALL_DATA_SIZE 102416306
#define MODEL_TYPE "SVD_BIAS"

// OVERVIEW 


/* Generally, we will be using the '../mu/all.dta' file. Moreover, as far as 
 * file structure goes, inside the same directory that this repository is 
 * stored in, include the um and mu files. In the future when we read in and
 * out using binary files rather than from the .dta Stata files, we will be 
 * reading and writing from a folder called binary_files. Include this folder
 * as well in the same directory as this repository. */


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
vector<vector<vector<double> > > get_training_data(int PERCENT) {
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

// Dot Product
inline double dot_product(double* A, double* B, int size) {
    double tot = 0.0;
    for (int i = 0; i < size; i++) 
        tot += A[i] * B[i];
    return tot;
}

inline double get_err(double** U, double** V, \
    double** Y, int K, int NUM_POINTS) {
    double err = 0.0;
    for (int i = 0; i < NUM_POINTS; i++) {
        int n = int(Y[i][0]) - 1;
        int m = int(Y[i][1]) - 1;
        double rating = Y[i][3];
        assert (rating <= 5.0);
        assert (rating >= 1.0);
        double predicted_rating = dot_product(U[n], V[m], K);
        if (predicted_rating < 1.0) {
            predicted_rating = 1.0;
        }
        else if (predicted_rating > 5.0) {
            predicted_rating = 5.0;
        }
        err += pow((predicted_rating - rating),2);
    }
    return pow(err/double(NUM_POINTS), 0.5);
}


// Takes the biases into account when calculating the error. 
inline double get_err_w_biases(double** U, double** V, \
    double** Y, int K, int NUM_POINTS, \
    double* USER_BIASES, double* MOVIE_BIASES, double average_rating) {

    double err = 0.0;
    double* point;
    for (int i = 0; i < NUM_POINTS; i++) {
        point = Y[i];
        int n = int(point[0]) - 1;
        int m = int(point[1]) - 1;
        double rating = point[3];
        // assert (rating <= 5.0);
        // assert (rating >= 1.0);
        double predicted_rating = dot_product(U[n], V[m], K) + USER_BIASES[n] \
                                        + MOVIE_BIASES[m] + average_rating;
        if (predicted_rating < 1.0) {
            predicted_rating = 1.0;
        }
        else if (predicted_rating > 5.0) {
            predicted_rating = 5.0;
        }
        err += pow((predicted_rating - rating),2);
    }
    return pow(err/double(NUM_POINTS), 0.5);
}


/* Creates the submission file */
inline void create_submission(double** U, double** V, \
    vector<vector<double> > test_set, int K, int NUM_TEST_POINTS, \
    double* USER_BIASES, double* MOVIE_BIASES, double average_rating, 
    double learning_rate, double regulation, int max_epochs, double final_eout, \
    int PERCENT) {
    vector<double> predicted_values;
    int n = 0; int m = 0; 
    double predicted_rating = 0.0;
    for (int i = 0; i < NUM_TEST_POINTS; i++) {
        vector<double> point = test_set[i];
        n = int(point[0]) - 1;
        m = int(point[1]) - 1;
        predicted_rating = dot_product(U[n], V[m], K) + USER_BIASES[n] + MOVIE_BIASES[m] + average_rating;
        if (predicted_rating < 1.0) {
            predicted_rating = 1.0;
        }
        else if (predicted_rating > 5.0) {
            predicted_rating = 5.0;
        }
        predicted_values.push_back(predicted_rating);
    }
    assert(predicted_values.size() == NUM_TEST_POINTS);
    string filename = create_submission_write_to_readme(MODEL_TYPE, learning_rate, regulation, max_epochs, final_eout, predicted_values, PERCENT, K);
    affirm_size_of_file(NUM_TEST_POINTS, filename);
}

/*
inline double get_err_preprocessing(double** Y, int mean, int NUM_POINTS) {
    double err = 0.0;
    for (int i = 0; i < NUM_POINTS; i++) {
        double rating = Y[i][3];
        assert (rating <= 5.0);
        assert (rating >= 1.0);
        double predicted_rating = mean;
        if (predicted_rating < 1.0) {
            predicted_rating = 1.0;
        }
        else if (predicted_rating > 5.0) {
            predicted_rating = 5.0;
        }
        err += pow((predicted_rating - rating),2);
    }
    return pow(err/double(NUM_POINTS), 0.5);

}
*/
inline void create_biases(double** TRAINING_DATA, double* USER_BIASES, double* MOVIE_BIASES, \
    double* user_counting, double* movie_counting, int NUM_POINTS, double average_rating) {
    int ri = 0;
    int ru = 0;
    int lambda_1 = 25;
    int lambda_2 = 10;
    double bi = 0.0; 
    double bu = 0.0;
    for (int j = 0; j < NUM_POINTS; j++) {
        movie_counting[int(TRAINING_DATA[j][1] - 1)] += 1;
        MOVIE_BIASES[int(TRAINING_DATA[j][1] - 1)] += TRAINING_DATA[j][3] - average_rating;
    }

    for (int i = 0; i < NUM_MOVIES; i++) {
        ri = movie_counting[i];
        bi = MOVIE_BIASES[i]/(lambda_1 + ri);
        MOVIE_BIASES[i] = bi;
    }
    for (int j = 0; j < NUM_POINTS; j++) {
        user_counting[int(TRAINING_DATA[j][0] - 1)] += 1;
        USER_BIASES[int(TRAINING_DATA[j][0] - 1)] += TRAINING_DATA[j][3] - average_rating - MOVIE_BIASES[int(TRAINING_DATA[j][1] - 1)];
    }
    for (int i = 0; i < NUM_USERS; i++) {
        ru = user_counting[i];
        bu = USER_BIASES[i]/(lambda_2 + ru);
        USER_BIASES[i] = bu;
    }
    /*
    for (int i = 0; i < NUM_USERS; i++) {
        int nu = user_counting[i];
        if (nu == 0) {
            theta = 0.0;
        }
        else {
            theta = double(USER_BIASES[i]) / nu;
        }
        double cu = (nu * theta) / (nu + alpha);

        USER_BIASES[i] = cu;
    } */

    /* double average_rating = 0.0;
    for (int i = 0; i < NUM_POINTS; i++) {
        average_rating += TRAINING_DATA[i][3];
    }
    average_rating /= NUM_POINTS;
    cout << "average_rating:  " << average_rating << endl;
    for (int i = 0; i < NUM_POINTS; i++) {
        MOVIE_BIASES[int(TRAINING_DATA[i][1]) - 1] += TRAINING_DATA[i][3] - average_rating;
        movie_counting[int(TRAINING_DATA[i][1]) - 1] += 1;
        USER_BIASES[int(TRAINING_DATA[i][0]) - 1] += TRAINING_DATA[i][3] - average_rating;
        user_counting[int(TRAINING_DATA[i][0]) - 1] += 1;
    } 
    cout << "Done here" << endl;
    for (int i = 0; i < NUM_MOVIES; i++) {
        if (movie_counting[i] == 0) {
            MOVIE_BIASES[i] = 0;
        }
        else {
            MOVIE_BIASES[i] = MOVIE_BIASES[i]/movie_counting[i];
        }
    }  
    for (int j = 0; j < NUM_USERS; j++) {
        if (user_counting[j] == 0) {
            USER_BIASES[j] = 0;
        }
        else {
            USER_BIASES[j] = USER_BIASES[j]/user_counting[j];
        }
    } 
    for (int i = 0; i < NUM_POINTS; i++) {
        int n = int(TRAINING_DATA[i][0]) - 1;
        int m = int(TRAINING_DATA[i][1]) - 1;
        TRAINING_DATA[i][3] -= (USER_BIASES[n] + MOVIE_BIASES[m] + average_rating);
    }
    */
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "usage: ./FILENAME   (N Percent Of Data)   (Eta)    (Reg)" << endl;
        return 0;
    }
    int PERCENT = stoi((argv[1]));
    double eta = stod((argv[2]));
    double reg = stod((argv[3]));
    double e_out;
    double* USER_BIASES = new double [NUM_USERS];
    double* user_counting = new double [NUM_USERS];
    double* MOVIE_BIASES = new double [NUM_MOVIES];
    double* movie_counting = new double [NUM_MOVIES];
    bool initialize = true;
    for (int i = 0; i < NUM_USERS; i++) {
        USER_BIASES[i] = 0;
        user_counting[i] = 0;
    }
    for (int j = 0; j < NUM_MOVIES; j++) {
        MOVIE_BIASES[j] = 0;
        movie_counting[j] = 0;
    }

    

    // First we load in the vector of all the training data. 
    vector<vector<vector<double> > > full_data = get_training_data(PERCENT);
    vector<vector<double> > training_data = full_data[0];
    vector<vector<double> > validation_data = full_data[3];
    /*
    vector<vector<double> > hidden_data = full_data[2];
    vector<vector<double> > probe_data = full_data[3];
    */
    vector<vector<double> > qual_data = full_data[4];



    // STEP 1: TRAINING DATA / VALIDATION ARRAY FILL IN. 
    /* We're gonna dynamically allocate memory here. BE VERY CAREFUL.
     * A MEMORY LEAK OF 2GB WILL FUCK UP YOUR COMPUTER. 
     */ 


    
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
    cout << "average_rating:  " << average_rating << endl;
    // Let's fill in the USER and MOVIE specific tables now. 
    if (initialize) {
        create_biases(TRAINING_DATA, USER_BIASES, MOVIE_BIASES, user_counting, movie_counting, NUM_POINTS, average_rating);
    }


    cout << "EVERYTHING IS DONE" << endl;
    // STEP 1: DONE 
    // WE WILL NOW TRAIN IT. 
    

    // STEP 2: INITIALIZE A LOT OF VALUES. INITIALIZE U AND V MATRICES. 
    // MOST CHANGES TO HYPERPARAMATERS WILL HAPPEN HERE. 
    double E_in_after; int x; int y; double Yij; double delta; 
    int M = NUM_USERS;
    int N = NUM_POINTS;
    int K = 200;
    int max_epochs = 50;
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
    // STEP 2: DONE 



    // STEP 3: Actual Training. 
    int size = NUM_POINTS;
    printf("learning rate = %f, training reg = %f, k = %d, M = %d, N = %d \n", \
        eta, reg, K, M, N); 
    double before_E_in; 
    double e_in;
    before_E_in = get_err_w_biases(U, V, TRAINING_DATA, K, NUM_POINTS, USER_BIASES, MOVIE_BIASES, average_rating);
    cout << "K:  " << K << endl;
    cout << "Max Epochs: " << max_epochs << endl;
    cout << "Initial Error:  " << before_E_in << "\n" << endl;
    double* point; double predicted_rating;
    int marker = int(size/100);
    double dot_prod;
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        before_E_in = get_err_w_biases(U, V, TRAINING_DATA, K, NUM_POINTS, USER_BIASES, MOVIE_BIASES, average_rating);
        for (int index = 0; index < size; index++) {
            point = TRAINING_DATA[index];
            x = point[0];
            y = point[1];
            Yij = point[3];
            dot_prod = dot_product(U[int(x) - 1], V[int(y) - 1], K); 
            predicted_rating = dot_prod + average_rating + USER_BIASES[int(x) - 1] + MOVIE_BIASES[int(y) - 1];
            e_in = Yij - predicted_rating;

            // Here we do the updates.

            // Update U Matrix.  
            for (int q = 0; q < K; q++) {
                U[int(x) - 1][q] = (U[int(x) - 1][q] * (1 - reg * eta)) + \
                    V[int(y) - 1][q] * (Yij - predicted_rating) * eta; 
            }

            // Update V Matrix. 
            // dot_prod = dot_product(U[int(x) - 1], V[int(y) - 1], K); 
            for (int q = 0; q < K; q++) {
                V[int(y) - 1][q] = (V[int(y) - 1][q] * (1 - reg * eta)) + \
                    U[int(x) - 1][q] * (Yij - predicted_rating) * eta; 
            }

            USER_BIASES[int(x) - 1] = USER_BIASES[int(x) - 1] +  eta * (e_in - reg * USER_BIASES[int(x) - 1]);
            MOVIE_BIASES[int(y) - 1] = MOVIE_BIASES[int(y) - 1] + eta * (e_in - reg * MOVIE_BIASES[int(y) - 1]);
            if (index % marker == 0) {
            // cout << counter << endl;
                cout.flush();
                cout << "\r" << "EPOCH " << epoch + 1 << ": " \
                    << index/marker  << " PERCENT COMPLETED   " ;
            }
        }
        if (epoch % 5 == 0) {
            E_in_after = get_err_w_biases(U,V,TRAINING_DATA, K, NUM_POINTS, USER_BIASES, MOVIE_BIASES, average_rating);
            e_out = get_err_w_biases(U, V, VALIDATION_DATA, K, NUM_VALIDATION, USER_BIASES, MOVIE_BIASES, average_rating);
  
            printf("E_in (RMSE): %f  ", E_in_after);
            printf("E_out (RMSE): %f ", e_out);
            printf("\n");
            if (epoch == 0) {
                delta = before_E_in - E_in_after;
            }
            else if (before_E_in - E_in_after < eps * delta) {
                break;
            }
        }
    } 
    
    // STEP 3: DONE. TRAINING Complete

    // LAST STEP: LET'S SEE WHAT OUR OUT OF SAMPLE ERROR IS GONNA BE. 
    e_out = get_err_w_biases(U, V, VALIDATION_DATA, K, NUM_VALIDATION, USER_BIASES, MOVIE_BIASES, average_rating);
    cout << "E_OUT: " << e_out << endl;


    // LAST STEP: DONE. Let's hope that was good enough. 


    // Let's create the submission file. 
    cout << "Test Set Size: " << qual_data.size() << endl; 
    create_submission(U, V, qual_data, K, qual_data.size(), USER_BIASES, MOVIE_BIASES, average_rating, eta, reg, max_epochs, e_out, PERCENT);
    // affirm_size(qual_data.size());

    // BUT WAIT THERE'S MORE. 
    // ABSOLUTELY CRITICAL THERE IS LITERALLY GIGABYTES OF DATA THAT NEEDS
    // TO BE FREED. 

    // Freeing memory. 
    
    for (int i = 0; i < NUM_POINTS; i++) {
        free(TRAINING_DATA[i]);
    }
    free(TRAINING_DATA);

    for (int i = 0; i < NUM_USERS; i++) {
        free(U[i]);
    }
    free(U);

    for (int i = 0; i < NUM_MOVIES; i++) {
        free(V[i]);
    }
    free(V);

    for (int i = 0; i < NUM_VALIDATION; i++) {
        free(VALIDATION_DATA[i]);
    }
    free(VALIDATION_DATA);
    free(USER_BIASES);
    free(MOVIE_BIASES);
    free(user_counting);
    free(movie_counting);
    return 1;

}



