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
using namespace std;
#define NUM_USERS 458293
#define NUM_MOVIES 17770
#define ALL_DATA_SIZE 102416306

/* ----------------------- General Comments and Logistics ---------------------

URLS:
    - http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf
    - http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/
    - http://eric-yuan.me/rbm/
    - https://deeplearning4j.org/restrictedboltzmannmachine



---------------------------------------------------------------------------- */

// ------------------------------ PREPROCESSING --------------------------------

std::vector<double> split_by_whitespace(string s) {
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
    afile.open("../mu/all.dta", ios::out | ios::in); 
    bfile.open("../mu/all.idx", ios::out | ios::in); 
    if (PERCENT == 100) {
        int one_percent = ALL_DATA_SIZE/100;
        while ((std::getline(afile, str)) && \
            (std::getline(bfile, identifier)))  {
            temp = split_by_whitespace(str);
            id = stod(identifier);
            if (temp[3] == 0) {
                assert (id == 5);
            }
            if (id == 1.0) {
                training_data.push_back(temp);
            }
            else if (id == 2.0) {
                validation_data.push_back(temp);
            }
            else if (id == 3.0) {
                hidden_data.push_back(temp);
            }
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
            if (id == 1.0) {
                training_data.push_back(temp);
            }
            else if (id == 2.0) {
                validation_data.push_back(temp);
            }
            else if (id == 3.0) {
                hidden_data.push_back(temp);
            }
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

// -------------------------------- TRAINGING CODE ------------------------------------

inline double logistic_fun (double a_i) {
    return 1 / (1 + exp(a_i));  
}

inline double activation_energy(double *weight){
    return 0.0;
} 






int main(int argc, char *argv[]) {
    
    // FIGURE OUT ARGS LATER ******************

    int PERCENT = 0.0;
    // First we load in the vector of all the training data. 
    vector<vector<vector<double> > > full_data = get_training_data(PERCENT);
    vector<vector<double> > training_data = full_data[0];
    vector<vector<double> > validation_data = full_data[1];
    vector<vector<double> > hidden_data = full_data[2];
    vector<vector<double> > probe_data = full_data[3];
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

    return 1; 

}
