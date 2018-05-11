
#include "ServerThread.hpp"
#include "SIMD-Scan/src/SIMD_Scan.cpp"
#include <stdexcept>

void Server::process( )
{
/* -- Something like this ...
    out.clear( );
    
    std::vector<bool> idx;
    scan(Table::getColumnA( ), jobs.in, idx);
    
    for(int i=0; i<idx.size(); ++i)
    {
        if (idx[i])
        {
            out.push_back(Table::get_A_at(i));
            out.push_back(Table::get_B_at(i));
        }
    }
*/
    
    throw std::runtime_error("Function Server::process has not been implemented, yet.");
}

void Server::stop( )
{
    throw std::runtime_error("Function Server::stop has not been implemented, yet.");
}
