# Parameter Server

## Parameter Server Architecture

![parallel\_computing\_client\_server](../.gitbook/assets/parallel_computing_client_server.png)

## The Parameter Server

* The parameter server was proposed by for scalable machine learning.
* Characters: client-server architecture, message-passing communication, and asynchronous.
* \(Note that MapReduce is bulk synchronous.\)
* Ray , an open-source software system, supports parameter server.

## Synchronous Algorithm

![parallel\_synchronous\_algorithm\_1](../.gitbook/assets/parallel_synchronous_algorithm_1.png)

## Asynchronous Algorithm

![parameter\_server\_architecture\_1](../.gitbook/assets/parameter_server_architecture_1.png)

## Parallel Asynchronous Gradient Descent

![parallel\_asynchronous\_gradient\_descent\_1](../.gitbook/assets/parallel_asynchronous_gradient_descent_1.png)

## Pro and Con of Asynchronous Algorithms

* In practice, asynchronous algorithms are faster than the synchronous.
* In theory, asynchronous algorithms has slower convergence rate.
* Asynchronous algorithms have restrictions, e.g., a worker cannot be much slower than the others.

![Pro\_and\_Con\_of\_Asynchronous\_Algorithms](../.gitbook/assets/Pro_and_Con_of_Asynchronous_Algorithms.png)

## Footnote

