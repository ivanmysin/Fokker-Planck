#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
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


#include <iostream>
#include <fstream>




int main(int argc, char **argv)
{

	brian_start();

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_neurongroup_clock_dt[0] = 0.0001;
        _array_neurongroup_clock_dt[0] = 0.0001;
        
                        
                        for(int i=0; i<_num__array_neurongroup_lastspike; i++)
                        {
                            _array_neurongroup_lastspike[i] = - INFINITY;
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_not_refractory; i++)
                        {
                            _array_neurongroup_not_refractory[i] = true;
                        }
                        
        _run_neurongroup_group_variable_set_conditional_codeobject();
        _run_synapses_synapses_create_generator_codeobject();
        _run_synapses_1_synapses_create_generator_codeobject();
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _array_neurongroup_clock_timestep[0] = 0;
        _array_neurongroup_clock_t[0] = 0.0;
        _run_synapses_1_pre_initialise_queue();
        _run_synapses_pre_initialise_queue();
        magicnetwork.clear();
        magicnetwork.add(&neurongroup_clock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&neurongroup_clock, _run_neurongroup_thresholder_codeobject);
        magicnetwork.add(&neurongroup_clock, _run_spikemonitor_codeobject);
        magicnetwork.add(&neurongroup_clock, _run_synapses_1_pre_push_spikes);
        magicnetwork.add(&neurongroup_clock, _run_synapses_1_pre_codeobject);
        magicnetwork.add(&neurongroup_clock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&neurongroup_clock, _run_synapses_pre_codeobject);
        magicnetwork.add(&neurongroup_clock, _run_neurongroup_resetter_codeobject);
        magicnetwork.add(&defaultclock, NULL);
        magicnetwork.run(1.0, NULL, 10.0);
        #ifdef DEBUG
        _debugmsg_spikemonitor_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_1_pre_codeobject();
        #endif

	}

	brian_end();

	return 0;
}