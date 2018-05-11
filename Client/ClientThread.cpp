#pragma once

#include "Client.hpp"

/* This function sets the vector requestedKeys */
void ClientThread::setKeys( )
{
    throw std::runtime_error("Function Server::process has not been implemented, yet.");
}

/* This function checks, whether the returned values are correct. Keep in mind, that the rows are stored column-wise, i.e. result = {A1, B1, A2, B2, ...} */
void ClientThread::checkReturnValues( )
{
/* --- Example implementation
    size_t numRows = returnedValues.size( ) / 2;
    
    for (int r=0; r<numRows; r+=2)
        if (returnedValues[r]*2 != returnedValues[r+1])
            throw std::invalid_arguments("Error: Incorrect Results!");
*/
    throw std::runtime_error("Function Server::process has not been implemented, yet.");
}

void ClientThread::stop( )
{
    repliedJobs = actuallyRepliedJobs;
    
    throw runtime_error("Function ClientThread::stop has not been implemented, yet.");
}
