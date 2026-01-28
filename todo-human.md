main(feature level)  
[ ] - discuss memory cli log with chat, --debug  
[ ] - sql performance log maybe  
[ ] - add e2e test for reader and memo  
[ ] - algo version control: move algo_libs to a new git repo  
[ ] - [cursor] set minimum functional requirements to force coding agent follow them while refactoring.  


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
[x] - [eval-dspy] explore to use dspy powered eval pipeline.  
[x] - [eval] merge the eval-dspy's heuristic rules into eval's heuristic rules.  
[x] - [eval] integrate new heuristic rules into run_eval.py and refactor run_eval.py.  
[ ] - [eval] test for new heuristic rules.  
[ ] - [reader,eval] port llm request logic out of eval to reader.  

[ ] - [chat,memo] check before implementing step4. do i need both topic_event and topic_lineage?  
[ ] - [reader,memo] implement step 4 - prompt user to choose one topic from step 3, and save to memo  

random(spikes, explore)  
[ ] - [reader] use lite llm and lite metadata(name, title, keywords) for summarization and thinking llm and depth-aware metadata for report genereation.  
