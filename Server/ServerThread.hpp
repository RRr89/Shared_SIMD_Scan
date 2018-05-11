#pragma once

#include <thread>
#include <stdexcept>

#include "utils/Network.hpp"

struct ServerPackage
{
    int server_id;
    int client_id;
    std::vector<int> in;
    std::vector<int> out;
}

class Server(int id)
{
private:
  ServerPackage job;
  bool isRunning;

public:
  Server(int id)
  : job( ), isRunning(false)
  {
    job.server_id = id;
    Network::register(id);
  }
  Server (const Server& other)
  : job( ), isRunning(false)
  {
      operator=(other);
  }
  const Server& operator=(const Server& other)
  {
      if (this->isRunning || other.isRunning)
          throw std::invalid_argument("Cannot copy running thread!");
      
      job.server_id = other.job.server_id;
      job.client_id = other.job.client_id;
      job.in = other.job.in;
      job.out = other.job.out;
      return other;
  }
  
  void run( )
  {
    isRunning = true;
    
    while (true)
    {
      Network::listen(job.server_id, &job.in, job.client_id);
    
      // do work
      process( );
    
      Network::send(job.server_id, job.client_id, job.out);
    }
    
    isRunning = false;
  }

  void stop( );
  void process( );
}
