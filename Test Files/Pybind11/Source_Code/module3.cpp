#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <vector>
using namespace std;


// See if any items in query are in the subject. Both must be 1D
bool array_check_1d(vector<int> query_vector, vector<int> subject_vector)
{
	bool overlap_bool;
	vector<int> overlap_vector;

	//first open up the subject.
	for (int i = 0; i < subject_vector.size(); ++i)
	{

		//Now open up the query.
		for (int q = 0; q < query_vector.size(); ++q)
		{
			if (query_vector[q] ==  subject_vector[i])
				overlap_bool = true;
		};
	};
	return overlap_bool;
}

// See if any items in query are in the subject. Both must be bigger than 2D
bool array_check_2d(vector<vector<int>> query_vector, vector<vector<int>> subject_vector)
{
	bool overlap_bool;
	vector<int> subject_item;

	//first open up the subject.
	for (int i = 0; i < subject_vector.size(); ++i)
	{

		//Now open up the query.
		for (int q = 0; q < query_vector.size(); ++q)
		{
			if (query_vector[q] ==  subject_vector[i])
				overlap_bool = true;
		};
	};
	return overlap_bool;
}


PYBIND11_MODULE (Array_Overlap, module)
{   // optional module docstring
    module.doc () = "Array Check";
    // define add function
    module.def("array_check_1d", &array_check_1d, "Check 2 1D arrays");
    module.def("array_check_2d", &array_check_2d, "Check 2 2D arrays");

}