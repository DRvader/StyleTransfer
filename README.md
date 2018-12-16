# StyleTransfer

A small project that I developed to build my understanding of the lowest level tensorflow python API.

Style Transfer was chosen because I wanted use it for another project and while I was looking for a good GAN implementation of it, 
I noticed that most tutorials that teach the original technique uses it to demonstrate eager execution. It got me thinking about doing
it via a static graph.

I initailly developed the simple_model first and was so interested by the basic graph traversal that I had to do, to get the feature 
layers than I did a more complex version in which I built a seperate graph to learn more about the Graph API.
