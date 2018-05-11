
#include <thread>
#include <iostream>
#include <chrono>

#include "Server/ServerThread.hpp"
#include "Client/ClientThread.hpp"
#include "utils/Environment.hpp"

size_t getNumServerThreads(int argc, char* argv);
size_t getNumClientThreads(int argc, char* argv);
size_t getBenchmarkExecutionTime(int argc, char* argv);

int main(int argc, char* argc)
{
    // prepare
    std::string filename = getTableLocation(argc, argv);
    Table::load(filename);
    
    Environment::ServerThreads = getNumServerThreads(argc, argv);
    Environment::ClientThreads = getNumServerThreads(argc, argv);
    
    std::vector<ServerThread*> servers;
    std::vector<ClientThread*> clients;
    std::vector<std::thread> threads;
    
    // start
    size_t i;
    for (i=0; i<Environment::ServerThreads; ++i)
        servers.push_back(new ServerThread(i));
    for (; i<Environment::ServerThreads + Environment::ClientThreads; ++i)
        clients.push_back(new ClientThread(i));
    
    for (size_t s=0; s<Environment::ServerThreads; ++s)
        threads.push_back(std::thread(&ServerThread::run, std::ref(servers[i])));
    
    size_t benchmarkExecutionTimeInSeconds = getBenchmarkExecutionTime(argc, argv);
    
    // execute
    for (size_t c=0; c<Environment::ClientThreads; ++c)
        threads.push_back(std::thread(ClientThread::run, std::ref(*clients[c])));
    
    std::this_thread::sleep_for (std::chrono::seconds(benchmarkExecutionTimeInSeconds));
    
    // stop
    for (size_t c=0; c<Environment::ClientThreads; ++c)
        clients[c]->stop( );
    for (size_t s=0; s<Environment::ServerThreads; ++s)
        servers[s].stop( );
    for (size_t t=0; t<threads.size( ); ++t)
        threads[t].join( );
    
    // analyze
    unsigned int numProcessedJobs = 0;
    for (size_t c=0; c<Environment::ClientThreads; ++c)
        numProcessedJobs += clients[c]->numProcessedJobs( );
    std::cout << "Environment: " << Environment::ServerThreads << " Servers and " << Environment::ClientThreads << " Clients." << std::endl;
    std::cout << "  Execution time   : " << benchmarkExecutionTimeInSeconds << " seconds." << std::endl;
    std::cout << "  Executed Requests: " << numProcessedJobs << " (in sum)" << std::endl;
    std::cout << "  Throughput       : " << (numProcessedJobs / static_cast<float>(benchmarkExecutionTimeInSeconds)) << " QPS" << std::endl;
    
    return 0;
}
