main(feature level)  
[ ] - discuss memory cli log with chat, --debug  
[ ] - sql performance log maybe  
[ ] - add e2e test for reader and memo  
[ ] - move algo_libs to a new git repo  


sub(issue, bug level)  
[x] - [chat] finalize the reader_algos package skeleton with chat.  
[x] - [reader] refactor for reader_algos package skeleton.  
[x] - [memo] refactor for reader_algos package skeleton.  
[x] - [reader] add reader output for step 1-2, refer to memory-cli/examples/  
[x] - [memo] add dry run mode  
[x] - [memo,reader] integration test for step 1-2  
[x] - [reader] refactor reader, make it scalable.  
[x] - [reader] manual test after refactor.  

[...] - [reader] implement step 3 - llm enrichment for monthly clusterings  
[x] - [eval] implement step 3 - setup eval pipeline for llm enrichment: metadata, prompt template.
[ ] - [eval] test eval pipeline.

[ ] - [chat,memo] check before implementing step4. do i need both topic_event and topic_lineage?  
[ ] - [reader,memo] implement step 4 - prompt user to choose one topic from step 3, and save to memo  
