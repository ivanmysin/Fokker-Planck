#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include "randomkit.h"

#include "code_objects/neurongroup_group_variable_set_conditional_codeobject.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_initialise_queue.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
    brian::neurongroup_clock.timestep = brian::_array_neurongroup_clock_timestep;
    brian::neurongroup_clock.dt = brian::_array_neurongroup_clock_dt;
    brian::neurongroup_clock.t = brian::_array_neurongroup_clock_t;
    for (int i=0; i<1; i++)
	    rk_randomseed(brian::_mersenne_twister_states[i]);  // Note that this seed can be potentially replaced in main.cpp
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


