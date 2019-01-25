#pragma once

#include <vector>
#include <iostream>
#include <fstream>


bool read_ext_file(char* filename, std::vector<float>& R, std::vector<float>& T, std::vector<float>& cam_params)
{
	auto is_valid = false;
	
	R = (3, std::vector<float>(3, 0));
	T = (3, std::vector<float>(1, 0));
	cam_params = (3, std::vector<float>(3, 0));





	return is_valid;
}