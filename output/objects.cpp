
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
Network magicnetwork;

//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup__spikespace;
const int _num__array_neurongroup__spikespace = 1001;
double * _array_neurongroup_clock_dt;
const int _num__array_neurongroup_clock_dt = 1;
double * _array_neurongroup_clock_t;
const int _num__array_neurongroup_clock_t = 1;
int64_t * _array_neurongroup_clock_timestep;
const int _num__array_neurongroup_clock_timestep = 1;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 1000;
double * _array_neurongroup_lastspike;
const int _num__array_neurongroup_lastspike = 1000;
double * _array_neurongroup_muext;
const int _num__array_neurongroup_muext = 1000;
char * _array_neurongroup_not_refractory;
const int _num__array_neurongroup_not_refractory = 1000;
int32_t * _array_neurongroup_subgroup_1__sub_idx;
const int _num__array_neurongroup_subgroup_1__sub_idx = 200;
int32_t * _array_neurongroup_subgroup__sub_idx;
const int _num__array_neurongroup_subgroup__sub_idx = 800;
double * _array_neurongroup_V;
const int _num__array_neurongroup_V = 1000;
int32_t * _array_spikemonitor__source_idx;
const int _num__array_spikemonitor__source_idx = 1000;
int32_t * _array_spikemonitor_count;
const int _num__array_spikemonitor_count = 1000;
int32_t * _array_spikemonitor_N;
const int _num__array_spikemonitor_N = 1;
int32_t * _array_synapses_1_N;
const int _num__array_synapses_1_N = 1;
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_spikemonitor_i;
std::vector<double> _dynamic_array_spikemonitor_t;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
std::vector<double> _dynamic_array_synapses_1_delay;
std::vector<double> _dynamic_array_synapses_1_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<double> _dynamic_array_synapses_lastupdate;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////

//////////////// synapses /////////////////
// synapses
SynapticPathway<double> synapses_pre(
		_dynamic_array_synapses_delay,
		_dynamic_array_synapses__synaptic_pre,
		0, 800);
// synapses_1
SynapticPathway<double> synapses_1_pre(
		_dynamic_array_synapses_1_delay,
		_dynamic_array_synapses_1__synaptic_pre,
		800, 1000);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp
Clock neurongroup_clock;  // attributes will be set in run.cpp

// Profiling information for each code object
double neurongroup_group_variable_set_conditional_codeobject_profiling_info = 0.0;
double neurongroup_resetter_codeobject_profiling_info = 0.0;
double neurongroup_stateupdater_codeobject_profiling_info = 0.0;
double neurongroup_thresholder_codeobject_profiling_info = 0.0;
double spikemonitor_codeobject_profiling_info = 0.0;
double synapses_1_pre_codeobject_profiling_info = 0.0;
double synapses_1_pre_initialise_queue_profiling_info = 0.0;
double synapses_1_pre_push_spikes_profiling_info = 0.0;
double synapses_1_synapses_create_generator_codeobject_profiling_info = 0.0;
double synapses_pre_codeobject_profiling_info = 0.0;
double synapses_pre_initialise_queue_profiling_info = 0.0;
double synapses_pre_push_spikes_profiling_info = 0.0;
double synapses_synapses_create_generator_codeobject_profiling_info = 0.0;

}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_neurongroup__spikespace = new int32_t[1001];
    
	for(int i=0; i<1001; i++) _array_neurongroup__spikespace[i] = 0;

	_array_neurongroup_clock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_neurongroup_clock_dt[i] = 0;

	_array_neurongroup_clock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_neurongroup_clock_t[i] = 0;

	_array_neurongroup_clock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_neurongroup_clock_timestep[i] = 0;

	_array_neurongroup_i = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_i[i] = 0;

	_array_neurongroup_lastspike = new double[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_lastspike[i] = 0;

	_array_neurongroup_muext = new double[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_muext[i] = 0;

	_array_neurongroup_not_refractory = new char[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_not_refractory[i] = 0;

	_array_neurongroup_subgroup_1__sub_idx = new int32_t[200];
    
	for(int i=0; i<200; i++) _array_neurongroup_subgroup_1__sub_idx[i] = 0;

	_array_neurongroup_subgroup__sub_idx = new int32_t[800];
    
	for(int i=0; i<800; i++) _array_neurongroup_subgroup__sub_idx[i] = 0;

	_array_neurongroup_V = new double[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_V[i] = 0;

	_array_spikemonitor__source_idx = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_spikemonitor__source_idx[i] = 0;

	_array_spikemonitor_count = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_spikemonitor_count[i] = 0;

	_array_spikemonitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_spikemonitor_N[i] = 0;

	_array_synapses_1_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;

	_array_synapses_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_synapses_N[i] = 0;


	// Arrays initialized to an "arange"
	_array_neurongroup_i = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_neurongroup_i[i] = 0 + i;

	_array_neurongroup_subgroup_1__sub_idx = new int32_t[200];
    
	for(int i=0; i<200; i++) _array_neurongroup_subgroup_1__sub_idx[i] = 800 + i;

	_array_neurongroup_subgroup__sub_idx = new int32_t[800];
    
	for(int i=0; i<800; i++) _array_neurongroup_subgroup__sub_idx[i] = 0 + i;

	_array_spikemonitor__source_idx = new int32_t[1000];
    
	for(int i=0; i<1000; i++) _array_spikemonitor__source_idx[i] = 0 + i;


	// static arrays

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_-8405552623266618253", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_-5625515268189269365", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_-8687466544330222760", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace_5193959326880884613", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 1001*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_clock_dt;
	outfile__array_neurongroup_clock_dt.open("results/_array_neurongroup_clock_dt_8699582524692276815", ios::binary | ios::out);
	if(outfile__array_neurongroup_clock_dt.is_open())
	{
		outfile__array_neurongroup_clock_dt.write(reinterpret_cast<char*>(_array_neurongroup_clock_dt), 1*sizeof(_array_neurongroup_clock_dt[0]));
		outfile__array_neurongroup_clock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_clock_dt." << endl;
	}
	ofstream outfile__array_neurongroup_clock_t;
	outfile__array_neurongroup_clock_t.open("results/_array_neurongroup_clock_t_-8718209519360606787", ios::binary | ios::out);
	if(outfile__array_neurongroup_clock_t.is_open())
	{
		outfile__array_neurongroup_clock_t.write(reinterpret_cast<char*>(_array_neurongroup_clock_t), 1*sizeof(_array_neurongroup_clock_t[0]));
		outfile__array_neurongroup_clock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_clock_t." << endl;
	}
	ofstream outfile__array_neurongroup_clock_timestep;
	outfile__array_neurongroup_clock_timestep.open("results/_array_neurongroup_clock_timestep_448900567270536520", ios::binary | ios::out);
	if(outfile__array_neurongroup_clock_timestep.is_open())
	{
		outfile__array_neurongroup_clock_timestep.write(reinterpret_cast<char*>(_array_neurongroup_clock_timestep), 1*sizeof(_array_neurongroup_clock_timestep[0]));
		outfile__array_neurongroup_clock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_clock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i_-1631232536865991351", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 1000*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_lastspike;
	outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike_-2206411325882622725", ios::binary | ios::out);
	if(outfile__array_neurongroup_lastspike.is_open())
	{
		outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 1000*sizeof(_array_neurongroup_lastspike[0]));
		outfile__array_neurongroup_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_muext;
	outfile__array_neurongroup_muext.open("results/_array_neurongroup_muext_-501118784543813554", ios::binary | ios::out);
	if(outfile__array_neurongroup_muext.is_open())
	{
		outfile__array_neurongroup_muext.write(reinterpret_cast<char*>(_array_neurongroup_muext), 1000*sizeof(_array_neurongroup_muext[0]));
		outfile__array_neurongroup_muext.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_muext." << endl;
	}
	ofstream outfile__array_neurongroup_not_refractory;
	outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory_7286082832221954493", ios::binary | ios::out);
	if(outfile__array_neurongroup_not_refractory.is_open())
	{
		outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 1000*sizeof(_array_neurongroup_not_refractory[0]));
		outfile__array_neurongroup_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_subgroup_1__sub_idx;
	outfile__array_neurongroup_subgroup_1__sub_idx.open("results/_array_neurongroup_subgroup_1__sub_idx_2612295432984313302", ios::binary | ios::out);
	if(outfile__array_neurongroup_subgroup_1__sub_idx.is_open())
	{
		outfile__array_neurongroup_subgroup_1__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_subgroup_1__sub_idx), 200*sizeof(_array_neurongroup_subgroup_1__sub_idx[0]));
		outfile__array_neurongroup_subgroup_1__sub_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_subgroup_1__sub_idx." << endl;
	}
	ofstream outfile__array_neurongroup_subgroup__sub_idx;
	outfile__array_neurongroup_subgroup__sub_idx.open("results/_array_neurongroup_subgroup__sub_idx_-3234139342938795343", ios::binary | ios::out);
	if(outfile__array_neurongroup_subgroup__sub_idx.is_open())
	{
		outfile__array_neurongroup_subgroup__sub_idx.write(reinterpret_cast<char*>(_array_neurongroup_subgroup__sub_idx), 800*sizeof(_array_neurongroup_subgroup__sub_idx[0]));
		outfile__array_neurongroup_subgroup__sub_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_subgroup__sub_idx." << endl;
	}
	ofstream outfile__array_neurongroup_V;
	outfile__array_neurongroup_V.open("results/_array_neurongroup_V_-2038475002070892598", ios::binary | ios::out);
	if(outfile__array_neurongroup_V.is_open())
	{
		outfile__array_neurongroup_V.write(reinterpret_cast<char*>(_array_neurongroup_V), 1000*sizeof(_array_neurongroup_V[0]));
		outfile__array_neurongroup_V.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_V." << endl;
	}
	ofstream outfile__array_spikemonitor__source_idx;
	outfile__array_spikemonitor__source_idx.open("results/_array_spikemonitor__source_idx_5666684372756098349", ios::binary | ios::out);
	if(outfile__array_spikemonitor__source_idx.is_open())
	{
		outfile__array_spikemonitor__source_idx.write(reinterpret_cast<char*>(_array_spikemonitor__source_idx), 1000*sizeof(_array_spikemonitor__source_idx[0]));
		outfile__array_spikemonitor__source_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__source_idx." << endl;
	}
	ofstream outfile__array_spikemonitor_count;
	outfile__array_spikemonitor_count.open("results/_array_spikemonitor_count_8050630968322604974", ios::binary | ios::out);
	if(outfile__array_spikemonitor_count.is_open())
	{
		outfile__array_spikemonitor_count.write(reinterpret_cast<char*>(_array_spikemonitor_count), 1000*sizeof(_array_spikemonitor_count[0]));
		outfile__array_spikemonitor_count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_count." << endl;
	}
	ofstream outfile__array_spikemonitor_N;
	outfile__array_spikemonitor_N.open("results/_array_spikemonitor_N_-8984066920862205899", ios::binary | ios::out);
	if(outfile__array_spikemonitor_N.is_open())
	{
		outfile__array_spikemonitor_N.write(reinterpret_cast<char*>(_array_spikemonitor_N), 1*sizeof(_array_spikemonitor_N[0]));
		outfile__array_spikemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor_N." << endl;
	}
	ofstream outfile__array_synapses_1_N;
	outfile__array_synapses_1_N.open("results/_array_synapses_1_N_5680748879185087668", ios::binary | ios::out);
	if(outfile__array_synapses_1_N.is_open())
	{
		outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(_array_synapses_1_N[0]));
		outfile__array_synapses_1_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_1_N." << endl;
	}
	ofstream outfile__array_synapses_N;
	outfile__array_synapses_N.open("results/_array_synapses_N_5340539973114803720", ios::binary | ios::out);
	if(outfile__array_synapses_N.is_open())
	{
		outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
		outfile__array_synapses_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N." << endl;
	}

	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i_-7739086547647816439", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
        if (! _dynamic_array_spikemonitor_i.empty() )
        {
			outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
		    outfile__dynamic_array_spikemonitor_i.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t_-3540577976695502271", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
        if (! _dynamic_array_spikemonitor_t.empty() )
        {
			outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
		    outfile__dynamic_array_spikemonitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_post;
	outfile__dynamic_array_synapses_1__synaptic_post.open("results/_dynamic_array_synapses_1__synaptic_post_1306176085068523458", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_post[0]), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(_dynamic_array_synapses_1__synaptic_post[0]));
		    outfile__dynamic_array_synapses_1__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
	outfile__dynamic_array_synapses_1__synaptic_pre.open("results/_dynamic_array_synapses_1__synaptic_pre_9081158206869443588", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses_1__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_pre[0]), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(_dynamic_array_synapses_1__synaptic_pre[0]));
		    outfile__dynamic_array_synapses_1__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_delay;
	outfile__dynamic_array_synapses_1_delay.open("results/_dynamic_array_synapses_1_delay_-3752321543681046269", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_delay.is_open())
	{
        if (! _dynamic_array_synapses_1_delay.empty() )
        {
			outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay[0]), _dynamic_array_synapses_1_delay.size()*sizeof(_dynamic_array_synapses_1_delay[0]));
		    outfile__dynamic_array_synapses_1_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_lastupdate;
	outfile__dynamic_array_synapses_1_lastupdate.open("results/_dynamic_array_synapses_1_lastupdate_-1272734881738961929", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_lastupdate.is_open())
	{
        if (! _dynamic_array_synapses_1_lastupdate.empty() )
        {
			outfile__dynamic_array_synapses_1_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_lastupdate[0]), _dynamic_array_synapses_1_lastupdate.size()*sizeof(_dynamic_array_synapses_1_lastupdate[0]));
		    outfile__dynamic_array_synapses_1_lastupdate.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_incoming;
	outfile__dynamic_array_synapses_1_N_incoming.open("results/_dynamic_array_synapses_1_N_incoming_-1916977233671612437", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_1_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_incoming[0]), _dynamic_array_synapses_1_N_incoming.size()*sizeof(_dynamic_array_synapses_1_N_incoming[0]));
		    outfile__dynamic_array_synapses_1_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_1_N_outgoing;
	outfile__dynamic_array_synapses_1_N_outgoing.open("results/_dynamic_array_synapses_1_N_outgoing_-4866843925531909690", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_1_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_outgoing[0]), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(_dynamic_array_synapses_1_N_outgoing[0]));
		    outfile__dynamic_array_synapses_1_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_post;
	outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post_9166908112804111266", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_post.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
		    outfile__dynamic_array_synapses__synaptic_post.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre_-7648332461290676713", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
			outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
		    outfile__dynamic_array_synapses__synaptic_pre.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_delay;
	outfile__dynamic_array_synapses_delay.open("results/_dynamic_array_synapses_delay_3782779011387385966", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_delay.is_open())
	{
        if (! _dynamic_array_synapses_delay.empty() )
        {
			outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
		    outfile__dynamic_array_synapses_delay.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_lastupdate;
	outfile__dynamic_array_synapses_lastupdate.open("results/_dynamic_array_synapses_lastupdate_8290347927056884214", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_lastupdate.is_open())
	{
        if (! _dynamic_array_synapses_lastupdate.empty() )
        {
			outfile__dynamic_array_synapses_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_lastupdate[0]), _dynamic_array_synapses_lastupdate.size()*sizeof(_dynamic_array_synapses_lastupdate[0]));
		    outfile__dynamic_array_synapses_lastupdate.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_incoming;
	outfile__dynamic_array_synapses_N_incoming.open("results/_dynamic_array_synapses_N_incoming_3938318868766884306", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_incoming.is_open())
	{
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
			outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
		    outfile__dynamic_array_synapses_N_incoming.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
	}
	ofstream outfile__dynamic_array_synapses_N_outgoing;
	outfile__dynamic_array_synapses_N_outgoing.open("results/_dynamic_array_synapses_N_outgoing_3054040452460589068", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_N_outgoing.is_open())
	{
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
			outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
		    outfile__dynamic_array_synapses_N_outgoing.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
	}


	// Write profiling info to disk
	ofstream outfile_profiling_info;
	outfile_profiling_info.open("results/profiling_info.txt", ios::out);
	if(outfile_profiling_info.is_open())
	{
	outfile_profiling_info << "neurongroup_group_variable_set_conditional_codeobject\t" << neurongroup_group_variable_set_conditional_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "neurongroup_resetter_codeobject\t" << neurongroup_resetter_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "neurongroup_stateupdater_codeobject\t" << neurongroup_stateupdater_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "neurongroup_thresholder_codeobject\t" << neurongroup_thresholder_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "spikemonitor_codeobject\t" << spikemonitor_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_pre_codeobject\t" << synapses_1_pre_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_pre_initialise_queue\t" << synapses_1_pre_initialise_queue_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_pre_push_spikes\t" << synapses_1_pre_push_spikes_profiling_info << std::endl;
	outfile_profiling_info << "synapses_1_synapses_create_generator_codeobject\t" << synapses_1_synapses_create_generator_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_pre_codeobject\t" << synapses_pre_codeobject_profiling_info << std::endl;
	outfile_profiling_info << "synapses_pre_initialise_queue\t" << synapses_pre_initialise_queue_profiling_info << std::endl;
	outfile_profiling_info << "synapses_pre_push_spikes\t" << synapses_pre_push_spikes_profiling_info << std::endl;
	outfile_profiling_info << "synapses_synapses_create_generator_codeobject\t" << synapses_synapses_create_generator_codeobject_profiling_info << std::endl;
	outfile_profiling_info.close();
	} else
	{
	    std::cout << "Error writing profiling info to file." << std::endl;
	}

	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
}

void _dealloc_arrays()
{
	using namespace brian;


	// static arrays
}

