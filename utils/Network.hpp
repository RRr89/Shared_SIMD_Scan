#pragma once

#include <vector>

#include "utils/concurrent_queue.hpp"

struct Message
{
    Message(const int from, const int to, const std::vector<int>& content)
    : source(from), destination(to), content(&content)
    { }
    Message( )
    : source(-1), destination(-1), content(nullptr)
    { }
    Message(const Message& other)
    : source(other.source), destination(other.destination), content(other.content)
    { }
    const Message& operator=(const Message& other)
    {
        source = other.source;
        destination = other.destination;
        content = other.content;
        return other;
    }
    
    int source;
    int destination;
    std::vector<int> const * content;
};

class Network
{
private:
  static std::map<int, concurrent_queue<Message>& > ThreadQueueRelation;
  static mutable boost::mutex the_mutex;
  
public:
  static void listen(const int port, std::vector<int>& content, int& sender)
  {
      Message m;
      ThreadQueueRelation[port].wait_and_pop(m);
      
      sender = m.source;
      content = *m.content;
  }
  
  static void send(const int from, const int to, const std::vector<int>& content)
  {
      Message m(from, to, content);
      ThreadQueueRelation[to].push(m);
  }
  
  static void register(const int port)
  {
      boost::mutex::scoped_lock lock(the_mutex);
      ThreadQueueRelation[port] = * new concurrent_queue<Message>( );
  }
};