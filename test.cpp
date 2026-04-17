#include <bits/stdc++.h>
using namespace std;

int main()
{
    int max_prob_size = 40;
    int probing_length = 1600;
    int max_value = 100;


    int current_prob_size = 8;
	int current_working_size = std::ceil((double)probing_length/current_prob_size);
	int diff = abs(current_prob_size - current_working_size);
	int best_prob_size = current_prob_size;
	for(int i = current_prob_size; i <= max_prob_size;i += 8)
	{
		current_working_size = std::ceil((double)max_value/i);
		if(abs(i - current_working_size) <= diff)
		{
			best_prob_size = i;
			diff = abs(i - current_working_size);
		}
        cout << current_working_size << " " << best_prob_size << " " << diff << endl;
	}
	cout << best_prob_size << endl;
    return 0;
}