#pragma once

/*This file holds various methods to output data at different stages for purposes of debugging*/

namespace lali {

class Debugger{
    void writeTimestreamToNetCDF();
};

Debugger::Debugger(void){
    SPDLOG_INFO("Created Debugger...")
}

void Debugger::writeTimestreamToNetCDF(){

}

} //namespace
