#pragma once

#include <vector>

#include "utils/Network.hpp"
#include "utils/Environment.hpp"

class ClientThread
{
private:
  int idx;
  unsigned int repliedJobs;
  unsigned int actuallyRepliedJobs;
  std::vector<int> requestedKeys;
  std::vector<int> returnedValues;
  
  /* This function sets the vector requestedKeys */
  void setKeys( );
  /* This function checks, whether the returned values are correct. Keep in mind, that the rows are stored column-wise, i.e. result = {A1, B1, A2, B2, ...} */
  void checkReturnValues( );
  
public:
  ClientThread(int id)
  : idx(id),repliedJobs(0), actuallyRepliedJobs(0), requestedKeys( ), returnedValues( )
  {
      Network::register(idx);
  }
  
  void run( )
  {
      actuallyRepliedJobs = 0;
      int dest = idx % Environment::ServerThreads;
      int replier;
      
      while (true)
      {
          setKeys( );
          
          Network::send(idx, dest, requestedKeys);
          Network::listen(idx, returnedValues, replier);
          
          checkReturnValues( );
          ++actuallyRepliedJobs;
      }
  }
  
  void stop( );
  
  unsigned int numProcessedJobs( )
  {
      return repliedJobs;
  }
  
  void startBenchmark( )
  {
      repliedJobs = 0;
  }
};
