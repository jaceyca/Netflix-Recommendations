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

#define WATER 0.95143

inline string create_submission_write_to_readme(string MODEL_TYPE, \
    double learning_rate, double regulation, int max_epoch, \
    double final_eout, vector<double> values, int PERCENT, int K) {
    double predicted_rating = 0.0;
    string filename = "../../submissions/";
    // string filename = "";
    filename += MODEL_TYPE;
    filename += "_lrt-";
    filename += to_string(learning_rate);
    filename += "_reg-";
    filename += to_string(regulation);
    filename += "_mepoch-"; 
    filename += to_string(max_epoch);
    filename += "_eout-";
    filename += to_string(final_eout);
    filename += ".txt";
    ofstream write_to_file; 
    ofstream write_to_readme;

    write_to_file.open(filename, std::ofstream::out | std::ofstream::trunc);
    write_to_readme.open("../README.md", std::ios::app);
    for (int i = 0; i < values.size(); i++) {
        predicted_rating = values[i];
        if (predicted_rating < 1.0) {
            predicted_rating = 1.0;
        }
        else if (predicted_rating > 5.0) {
            predicted_rating = 5.0;
        }
        write_to_file << predicted_rating << '\n';
    }
    write_to_readme << MODEL_TYPE + " " + to_string(PERCENT) + " " + to_string(K) + " " + to_string(learning_rate) + " " + to_string(regulation) + " " + to_string(max_epoch) + " " + to_string((WATER - final_eout)/100) + "\n";
    write_to_file.close();
    write_to_readme.close();
    return filename;

}

inline void affirm_size_of_file(int test_size, string filename) {
    fstream afile; string str; int counter = 0;
    afile.open(filename, ios::out | ios::in); 
    while (std::getline(afile, str)) {
        counter += 1;
    }
    afile.close();
    assert (counter == test_size);
}

