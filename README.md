# Artificial-intelligence-chatbot

## Summary

The architecture intended for the system is sequence-to-sequence architecture. To execute, you first need to convert the sentences into code, the conversion of sentences into code is done by a dictionary that is a word in the database. The network of choice for the system is the LSTM network, and since the input of this network must be a vector, the words that were converted to code in the previous step enter an embedding layer to become a vector, these vectors are the input of the LSTM network Encoder .



![01](https://user-images.githubusercontent.com/76064876/142715107-6febcac6-e5ef-493a-92be-23578124fbe8.jpg)
