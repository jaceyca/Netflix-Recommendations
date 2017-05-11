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
#include <random>
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

// -------------------------------- TRAINING CODE ------------------------------------

inline vector <vector <double> > logistic (vector <vector <double> > matrix) {
    vector <vector <double> > result = matrix;
    unsigned int matrix_rows = matrix.size();
    unsigned int matrix_cols = matrix[0].size();
    for (unsigned int i = 0; i < matrix_rows; i++) {
        for (unsigned int j = 0; j < matrix_cols; j++) {
            result[i][j] = 1.0 / (1.0 + exp(-(matrix[i][j])));
        }
    }
    return result;
}

inline vector <vector <double> > matrix_mult(vector <vector <double> > A, vector <vector <double> > B) {
    // A is a i x k matrix, B is a k x j matrix
    unsigned int A_rows = A.size(); // i
    unsigned int A_cols = A[0].size(); // k
    unsigned int B_rows = B.size(); // k
    unsigned int B_cols = B[0].size(); // j

    if (A_cols != B_rows) {
        cout << "A_cols: " << A_cols << ", B_rows: " << B_rows << " are not equal" << endl;
    }
    vector <vector <double> > result(A_rows, vector<double>(B_cols));    // i x j matrix
    for (unsigned int i = 0; i < A_rows; i++) {
        for (unsigned int j = 0; j < B_cols; j++) {
            result[i][j] = 0.0;
            for (unsigned int k = 0; k < B_rows; k++) {
                result[i][j] = result[i][j] + (A.at(i)).at(k) * (B.at(k)).at(j);
            }
        }
    }
    return result;
}

inline vector <vector <double> > mm_transpose(vector <vector <double> > A, vector <vector <double> > B) {
    // A is a k x i matrix, B is a k x j matrix
    unsigned int A_rows = A.size(); // k
    unsigned int A_cols = A[0].size(); // i
    unsigned int B_rows = B.size(); // k
    unsigned int B_cols = B[0].size(); // j

    if (A_rows != B_rows) {
        cout << "A_rows: " << A_rows << ", B_rows: " << B_rows << " are not equal" << endl;
    }
    vector <vector <double> > result(A_cols, vector<double>(B_cols));    // i x j matrix
    for (unsigned int i = 0; i < A_cols; i++) {
        for (unsigned int j = 0; j < B_cols; j++) {
            result[i][j] = 0.0;
            for (unsigned int k = 0; k < B_rows; k++) {
                result[i][j] = result[i][j] + (A.at(k)).at(i) * (B.at(k)).at(j);
            }
        }
    }
    return result;
}

inline vector <vector <double> > mm_transpose2(vector <vector <double> > A, vector <vector <double> > B) {
    // A is a i x k matrix, B is a j x k matrix
    unsigned int A_rows = A.size(); // i
    unsigned int A_cols = A[0].size(); // k
    unsigned int B_rows = B.size(); // j
    unsigned int B_cols = B[0].size(); // k

    if (A_cols != B_cols) {
        cout << "A_cols: " << A_cols << ", B_cols: " << B_cols << " are not equal" << endl;
    }
    vector <vector <double> > result(A_rows, vector<double>(B_rows));    // i x j matrix
    for (unsigned int i = 0; i < A_rows; i++) {
        for (unsigned int j = 0; j < B_rows; j++) {
            result[i][j] = 0.0;
            for (unsigned int k = 0; k < B_cols; k++) {
                result[i][j] = result[i][j] + (A.at(i)).at(k) * (B.at(j)).at(k);
            }
        }
    }
    return result;
}


class RBM { 

     public: 
        RBM (int num_visible, int num_hidden, float learning_rate);
        vector<vector<double> > data;
        void train(vector<vector<double> > data, int epochs);
        vector<vector<double> > run_hidden(vector<vector<double> > data);
        vector<vector<double> > run_visible(vector<vector<double> > data);
		int num_visible;
		int num_hidden;
		float learning_rate;
		vector<vector<double> > weights;
};

// This is the constructor of all an RBM instance
RBM::RBM (int num_visible, int num_hidden, float learning_rate) {
        this->num_hidden = num_hidden + 1;
        this->num_visible = num_visible + 1;
        this->learning_rate = learning_rate; 
        
        weights = vector<vector<double> >((num_visible + 1), vector<double>(num_hidden + 1));
        for(int k = 0; k < num_hidden + 1; k++) {
            weights[0][k] = 0.0;
        }

        for(int l = 0; l < num_visible + 1; l++) {
            weights[l][0] = 0.0;
        }
        
        // normal distribution that we are going to use to fill up our weight matrix 
        normal_distribution<double> distribution(0.0, 1.0); 
        default_random_engine generator; 

        for(int i = 1; i < num_hidden + 1; i++){
            for(int j = 1; j < num_visible + 1; j++){
                weights[i][j] = 0.1 * distribution(generator);
            }
        }
}

// This is the training function 
void RBM::train(vector<vector<double> > data, int epochs) {
    // num_examples = data.shape[0]
    // data.size = # of rows, data[0].size = # of cols
    unsigned int num_examples = data.size() + 1;
    // note that num_examples = num_visible because we take the matrix_mult of data and weights
    vector <vector <double> > pos_hidden_activations(num_examples, vector <double>(num_hidden));
    vector <vector <double> > pos_hidden_probs(num_examples, vector <double>(num_hidden)); 
    vector <vector <double> > rand_hidden_states(num_examples, vector <double>(num_hidden));
    vector <vector <double> > pos_hidden_states(num_examples, vector <double>(num_hidden)); 
    vector <vector <double> > pos_associations(num_visible, vector <double>(num_hidden));
    vector <vector <double> > neg_visible_activations(num_examples, vector <double>(num_visible));  
    vector <vector <double> > neg_visible_probs(num_examples, vector <double>(num_visible)); 
    vector <vector <double> > neg_hidden_activations(num_examples, vector <double>(num_hidden));  
    vector <vector <double> > neg_hidden_probs(num_examples, vector <double>(num_hidden));  
    vector <vector <double> > neg_associations(num_visible, vector <double>(num_hidden));  
    // First we have to add a bias term into our data 
    vector <vector <double> > bias_data(num_examples, vector <double>(data[0].size() + 1));
    // cout << "weightstrain" << weights.size() << endl;
    // cout << "weightstrain[0]" << weights[0].size() << endl;
    // cout << "num_examples" << num_examples << endl;
    // cout << "num_visible" << num_visible << endl;
    // cout << "num_hidden" << num_hidden << endl;
    // now we must populate the bias term
    // data = np.insert(data, 0, 1, axis = 1)
    // insert 1s into the first column
    for(unsigned int k = 0; k < data.size(); k++) {
        bias_data[k][0] = 1.0;
    }

     // then we populate the rest of the the data with the bias data 
    for (unsigned int l = 0; l < data.size(); l++){
        for(unsigned int j = 1; j < data[0].size(); j++) {
            bias_data[l][j] = data[l][j];
            // bias_data.at(l).push_back(data[l][j]);
        }
    }
    // for epoch in range(max_epochs):      
    for(int e = 0; e < epochs; e++) {
        // pos_hidden_activations = np.dot(data, self.weights)      
        pos_hidden_activations = matrix_mult(bias_data, weights);
        // pos_hidden_probs = self._logistic(pos_hidden_activations)
        pos_hidden_probs = logistic(pos_hidden_activations);

        normal_distribution<double> distribution(0.0, 1.0); 
        default_random_engine generator; 
        // pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        for(unsigned int i = 0; i < num_examples; i++){
            for(int j = 0; j < num_hidden; j++){
                rand_hidden_states[i][j] = distribution(generator);
            }
        }

        for (unsigned int i = 0; i < num_examples; i++) {
            for (int j = 0; j < num_hidden; j++) {
                if (pos_hidden_probs[i][j] > rand_hidden_states[i][j]) {
                    pos_hidden_states[i][j] = 1.0;
                }
                else {
                    pos_hidden_states[i][j] = 0.0;
                }
            }
        }    
        // pos_associations = np.dot(data.T, pos_hidden_probs)
        pos_associations = mm_transpose(bias_data, pos_hidden_probs);
        // neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
        neg_visible_activations = mm_transpose2(pos_hidden_states, weights);
        // neg_visible_probs = self._logistic(neg_visible_activations)
        neg_visible_probs = logistic(neg_visible_activations);
        //  neg_visible_probs[:,0] = 1.   [:,0] means the first column
        for (unsigned int i = 0; i < num_examples; i++) {
           neg_visible_probs[i][0] = 1.0;
        }
        // neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
        neg_hidden_activations = matrix_mult(neg_visible_probs, weights);
        // neg_hidden_probs = self._logistic(neg_hidden_activations)
        neg_hidden_probs = logistic(neg_hidden_activations);
        // neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
        neg_associations = mm_transpose(neg_visible_probs, neg_hidden_probs);

        // self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)
        for (unsigned int i = 0; i < pos_associations.size(); i++) {
            for (unsigned int j = 0; j < pos_associations[0].size(); j++) {
                weights[i][j] += learning_rate * ((pos_associations[i][j] - neg_associations[i][j]) / num_examples);
            }
        }

        // error = np.sum((data - neg_visible_probs) ** 2)
        vector <vector <double> > error(num_examples, vector <double> (bias_data[0].size()));
        double error_sum = 0.0;
        for (unsigned int i = 0; i < num_examples; i++) {
           for (unsigned int j = 0; j < bias_data[0].size(); j++) {
               error[i][j] = bias_data[i][j] - neg_visible_probs[i][j];
               error_sum += ((error[i][j]) * (error[i][j]));
           }
        }
        // printf("Epoch is %d, error is %f \n", e, error_sum);
        cout << "Epoch is " << e << ", error is " << error_sum << endl;
    }

    return;
}

vector<vector<double> > RBM::run_visible(vector<vector<double> > data) {
    // num_examples = data.shape[0]
    unsigned int num_examples = data.size() + 1;
    vector<vector<double> > hidden_states(num_examples, vector<double>(num_hidden));
    vector<vector<double> > bias_data(num_examples, vector<double>(data[0].size() + 1));
    vector<vector<double> > hidden_activations(num_examples, vector<double>(num_hidden));
    vector<vector<double> > hidden_probs(num_examples, vector<double>(num_hidden));
    vector<vector<double> > rand_hidden_states(num_examples, vector<double>(num_hidden));
    // vector<vector<double> > new_hidden_states(num_examples, vector<double>(num_hidden - 1));


    // cout << "num_hid" << num_hidden << endl;
    // cout << "num_vis" << num_visible << endl;
    // cout << "num_exa" << num_examples << endl;
    
    // might be unnecessary because we use variable-sized arrays
    // hidden_states = np.ones((num_examples, self.num_hidden + 1))
    for (unsigned int i = 0; i < num_examples; i++) {
        for (int j = 0; j < num_hidden; j++) {
            hidden_states[i][j] = 1.0;
        }
    }

    // now we must populate the bias term
    // data = np.insert(data, 0, 1, axis = 1)
    // insert 1s into the first column
    for(unsigned int k = 0; k < data.size(); k++) {
        bias_data[k][0] = 1.0;
    }

     // then we populate the rest of the the data with the bias data 
    for (unsigned int l = 0; l < data.size(); l++){
        for(unsigned int j = 1; j < data[0].size(); j++) {
            bias_data[l][j] = data[l][j];
            // bias_data.at(l).push_back(data[l][j]);
        }
    }
    
    // hidden_activations = np.dot(data, self.weights)
    hidden_activations = matrix_mult(bias_data, weights);
    // hidden_probs = self._logistic(hidden_activations)
    hidden_probs = logistic(hidden_activations);

    normal_distribution<double> distribution(0.0, 1.0); 
    default_random_engine generator; 

    // hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    for(unsigned int i = 0; i < num_examples; i++){
        for(int j = 0; j < num_hidden; j++){
            rand_hidden_states[i][j] = distribution(generator);
        }
    }

    for (unsigned int i = 0; i < num_examples; i++) {
        for (int j = 0; j < num_hidden; j++) {
            if (hidden_probs[i][j] > rand_hidden_states[i][j]) {
                hidden_states[i][j] = 1.0;
            }
            else {
                hidden_states[i][j] = 0.0;
            }
        }
    }

    // hidden_states[:,0] = 1, fixing bias unit to 1
    for (unsigned int i = 0; i < hidden_states.size(); i++) {
       hidden_states[i][0] = 1.0;
    }

    // hidden_states = hidden_states[:,1:]
    // ignore the first column
    // for (unsigned int i = 0; i < hidden_states.size(); i++) {
    //     for (unsigned int j = 1; j < hidden_states[0].size(); j++) {
    //         new_hidden_states[i][j-1] = hidden_states[i][j];
    //     }
    // }

    return hidden_states;
}

vector<vector<double> > RBM::run_hidden(vector<vector<double> > data) {
    // num_examples = data.shape[0]
    unsigned int num_examples = data.size() + 1;
    vector<vector<double> > bias_data(num_examples, vector<double>(data[0].size() + 1));
    vector<vector<double> > visible_states(num_examples, vector<double>(num_hidden));
    vector<vector<double> > visible_activations(num_examples, vector<double>(num_hidden));
    vector<vector<double> > visible_probs(num_examples, vector<double>(num_hidden));
    vector<vector<double> > rand_visible_states(num_examples, vector<double>(num_hidden));
    vector<vector<double> > new_visible_states(num_examples, vector<double>(num_hidden - 1));

    // visible_states = np.ones((num_examples, self.num_visible + 1))
    for (unsigned int i = 0; i < num_examples; i++) {
        for (int j = 0; j < num_visible + 1; j++) {
            visible_states[i][j] = 1.0;
        }
    }

    // now we must populate the bias term
    // data = np.insert(data, 0, 1, axis = 1)
    // insert 1s into the first column
    for(unsigned int k = 0; k < data.size(); k++) {
        bias_data[k][0] = 1.0;
    }

     // then we populate the rest of the the data with the bias data 
    for (unsigned int l = 0; l < data.size(); l++){
        for(unsigned int j = 1; j < data[0].size(); j++) {
            bias_data[l][j] = data[l][j];
            // bias_data.at(l).push_back(data[l][j]);
        }
    }

    // visible_activations = np.dot(data, self.weights.T)
    visible_activations = matrix_mult(bias_data, weights);
    // visible_probs = self._logistic(visible_activations)
    visible_probs = logistic(visible_activations);

    normal_distribution<double> distribution(0.0, 1.0); 
    default_random_engine generator; 

    // visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    for(unsigned int i = 0; i < num_examples; i++){
        for(int j = 0; j < num_visible; j++){
            rand_visible_states[i][j] = distribution(generator);
        }
    }

    for (unsigned int i = 0; i < num_examples; i++) {
        for (int j = 0; j < num_visible; j++) {
            if (visible_probs[i][j] > rand_visible_states[i][j]) {
                visible_states[i][j] = 1.0;
            }
            else {
                visible_states[i][j] = 0.0;
            }
        }
    }

    // visible_states[:,0] = 1, fixing bias unit to 1
    for (unsigned int i = 0; i < visible_states.size(); i++) {
       visible_states[i][0] = 1.0;
    }

    // visible_states = visible_states[:,1:]
    // ignore the first column
    for (unsigned int i = 0; i < visible_states.size(); i++) {
        for (unsigned int j = 1; j < visible_states[0].size(); j++) {
            new_visible_states[i][j-1] = visible_states[i][j];
        }
    }

    return new_visible_states;
}

int main(int argc, char *argv[]) {
    // FIGURE OUT ARGS LATER ******************
    // r = RBM(num_visible = 6, num_hidden = 2, learning_rate = 0.1)
    RBM r = RBM(6, 2, 0.1);
    // training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],
    // [0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])

    vector<vector<double> > training_data {{1.0, 1.0, 1.0, 0.0, 0.0, 0.0}, {1.0,0.0,1.0,0.0,0.0,0.0}, \
    {1.0,1.0,1.0,0.0,0.0,0.0}, {0.0,0.0,1.0,1.0,1.0,0.0}, {0.0,0.0,1.0,1.0,0.0,0.0}, {0.0,0.0,1.0,1.0,1.0,0.0}};

    // r.train(training_data, max_epochs = 5000)
    r.train(training_data, 1000);
    // printf(r.weights);
    for (unsigned int i = 0; i < r.weights.size(); i++) {
        for (unsigned int j = 0; j < r.weights[0].size(); j++) {
            cout << "weights[" << i << "][" << j << "] = " << r.weights[i][j] << endl;
        }
    }
    // user = np.array([[0,0,0,1,1,0]])
    vector<vector<double> > user {{0,0,0,1,1,0}};
    // print(r.run_visible(user))
    vector<vector<double> > new_hidden_states2 = r.run_visible(user);
    cout << "new_hs: " << new_hidden_states2.size() << ", " << new_hidden_states2[0].size() << endl;
    for (unsigned int i = 0; i < new_hidden_states2.size(); i++) {
        for (unsigned int j = 0; j < new_hidden_states2[0].size(); j++) {
            cout << "run_visible[" << i << "][" << j << "] = " << new_hidden_states2[i][j] << endl;
        }
    }

    return 1; 

    // int PERCENT = 0.0;
    // // First we load in the vector of all the training data. 
    // vector<vector<vector<double> > > full_data = get_training_data(PERCENT);
    // vector<vector<double> > training_data = full_data[0];
    // vector<vector<double> > validation_data = full_data[1];
    // vector<vector<double> > hidden_data = full_data[2];
    // vector<vector<double> > probe_data = full_data[3];
    // vector<vector<double> > qual_data = full_data[4];



    // // STEP 1: TRAINING DATA / VALIDATION ARRAY FILL IN. 
    // /* We're gonna dynamically allocate memory here. BE VERY CAREFUL.
    //  * A MEMORY LEAK OF 2GB WILL FUCK UP YOUR COMPUTER. 
    //  */ 

    // int NUM_POINTS = training_data.size();
    // int POINT_WIDTH = training_data[0].size();
    // int NUM_VALIDATION = validation_data.size();
    // cout << "VALIDATON SET SIZE:   " << NUM_VALIDATION << endl;
    // cout << "TEST SET SIZE:   " << qual_data.size() << endl;
    // if (PERCENT == 100) {
    //     assert (qual_data.size() == 2749898);
    // }
    // // Setting up training data. 
    // double** TRAINING_DATA = new double* [NUM_POINTS];
    // for (int i = 0; i < NUM_POINTS; i++) {
    //     TRAINING_DATA[i] = new double[POINT_WIDTH];
    // }
    // for (int i = 0; i < NUM_POINTS; i++) {
    //     for (int j = 0; j < POINT_WIDTH; j++) {
    //         TRAINING_DATA[i][j] = training_data[i][j]; 
    //     }
    // }

    // // Setting up validation data. 
    // double** VALIDATION_DATA = new double* [NUM_VALIDATION];
    // for (int i = 0; i < NUM_VALIDATION; i++) {
    //     VALIDATION_DATA[i] = new double[POINT_WIDTH];
    // }
    // for (int i = 0; i < NUM_VALIDATION; i++) {
    //     for (int j = 0; j < POINT_WIDTH; j++) {
    //         VALIDATION_DATA[i][j] = validation_data[i][j]; 
    //     }
    // }
}
