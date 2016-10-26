# GPU Programming Assignment v0.8

by Jack Wadden

# Description
Assignment slides are located here: http://www.cs.virginia.edu/~jpw8bd/teaching/

Assignment text is located here: https://docs.google.com/document/d/1Ug4RTt0PnMTcYjAJ24qEeWWHaLanJ9NdY6cAQSSBA-w/edit?usp=sharing

This assignment is meant to teach undergraduate/graduate level programmers basic GPU programming by having them write a parallel findMax() reduction in the CUDA programming language. 

**Question 1** is meant to familiarize the student with host-side kernel invocation. 

**Question 2** is meant to introduce the idea of shared memory as a low-latency, software managed scratchpad. 

**Question 3** introduces thread level parallelism within thread-blocks. 

**Question 4** optimizes the solution to Question 3, highlighting the impact of control flow divergence within a SIMD instruction. 

NOTE: this is meant to follow a lecture introducing modern GPU architecture and the CUDA programming language.

# License
Copyright (c) 2016, John "Jack" Pierson Wadden

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
