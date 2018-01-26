
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>


namespace brian {

// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////
extern Clock defaultclock;
extern Clock neurongroup_clock;

//////////////// networks /////////////////
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_1_delay;
extern std::vector<double> _dynamic_array_synapses_1_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_delay;
extern std::vector<double> _dynamic_array_synapses_lastupdate;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern double *_array_neurongroup_clock_dt;
extern const int _num__array_neurongroup_clock_dt;
extern double *_array_neurongroup_clock_t;
extern const int _num__array_neurongroup_clock_t;
extern int64_t *_array_neurongroup_clock_timestep;
extern const int _num__array_neurongroup_clock_timestep;
extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double *_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;
extern double *_array_neurongroup_muext;
extern const int _num__array_neurongroup_muext;
extern char *_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;
extern int32_t *_array_neurongroup_subgroup_1__sub_idx;
extern const int _num__array_neurongroup_subgroup_1__sub_idx;
extern int32_t *_array_neurongroup_subgroup__sub_idx;
extern const int _num__array_neurongroup_subgroup__sub_idx;
extern double *_array_neurongroup_V;
extern const int _num__array_neurongroup_V;
extern int32_t *_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t *_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t *_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;
extern int32_t *_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t *_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////

//////////////// synapses /////////////////
// synapses
extern SynapticPathway<double> synapses_pre;
// synapses_1
extern SynapticPathway<double> synapses_1_pre;

// Profiling information for each code object
extern double neurongroup_group_variable_set_conditional_codeobject_profiling_info;
extern double neurongroup_resetter_codeobject_profiling_info;
extern double neurongroup_stateupdater_codeobject_profiling_info;
extern double neurongroup_thresholder_codeobject_profiling_info;
extern double spikemonitor_codeobject_profiling_info;
extern double synapses_1_pre_codeobject_profiling_info;
extern double synapses_1_pre_initialise_queue_profiling_info;
extern double synapses_1_pre_push_spikes_profiling_info;
extern double synapses_1_synapses_create_generator_codeobject_profiling_info;
extern double synapses_pre_codeobject_profiling_info;
extern double synapses_pre_initialise_queue_profiling_info;
extern double synapses_pre_push_spikes_profiling_info;
extern double synapses_synapses_create_generator_codeobject_profiling_info;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


