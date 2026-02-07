main(feature level)  
[ ] - discuss memory cli log with chat, --debug  
[ ] - sql performance log maybe  
[ ] - add e2e test for reader and memo  
[ ] - algo version control: move algo_libs to a new git repo  
[ ] - [cursor] set minimum functional requirements to force coding agent follow them while refactoring.  
[ ] - [eval] implement a llm request&response cache mechanism  
[ ] - [eval, reader] design a structure to share the llm evaluation logic between eval pipeline and reader. f.e use the llm heuristic rules and judge from reader.  
[ ] - [reader] **consider kmeans++?**  
[ ] - [memo] enforce machine-only design in logging system: diagnostic logs -> stderr, cmd output -> stdout  
[ ] - now we have tokenbucket per pipeline execution time. consider adding a second layer of tokenbucket for per llm provider: pipeline token bucket -> llm provider token bucket. and also consider caching them in db.  
[ ] - [memo] auto db migration. because the sqlite is designed to be a local db/memory system and is tighted to the memo cli. woould be interested to think of a way to handle local db migration after cli upgrade. a classical way just use a bunch of migration scripts. but since this is a machine-only/agent facing tool, maybe the migration can be some sort of prompt files and let agent drives/suggests the local db change vs new db schema.  

sub(issue, bug level)  
[x] - [chat] finalize the reader_algos package skeleton with chat.  
[x] - [reader] refactor for reader_algos package skeleton.  
[x] - [memo] refactor for reader_algos package skeleton.  
[x] - [reader] add reader output for step 1-2, refer to memory-cli/examples/  
[x] - [memo] add dry run mode  
[x] - [memo,reader] integration test for step 1-2  
[x] - [reader] refactor reader, make it scalable.  
[x] - [reader] manual test after refactor.  

 
[x] - [eval] implement step 3 - setup eval pipeline for llm enrichment: metadata, prompt template.  
[x] - [eval-dspy] explore to use dspy powered eval pipeline.  
[x] - [eval] merge the eval-dspy's heuristic rules into eval's heuristic rules.  
[x] - [eval] integrate new heuristic rules into run_eval.py and refactor run_eval.py.  
[x] - [eval] test for new heuristic rules.  

[x] - [reader,eval] port llm request logic out of eval to reader.  

[x] - [reader,memo] integration test for fetch, cluster+embed, memo fresh paper, memo get clusters, llm enrichment  
[x] - [reader-evolution-pip] draft design  
[x] - [reader,memo,chat?] llm-enrichment UX and db storage. edit: llm-enrichment UX is under tui/.  llm-enrichment db storage is adding new memo injest-cluster-observation cmd.  

[x] - [reader] implement llm-enrichment UX under pipelines/tui.  
[x] - [memo] add new ingest cluster observation(llm enrichment) cmd.  
[x] - [reader] integrate new memo ingest cluster observation cmd.  
[x] -  [reader,memo] test new memo ingest cluster observation cmd.  


[x] - [memo,reader] add new memo get-cluster-observation cmd.  
[x] - [reader] integrate tui into monthly pipeline.  


[x] - [chat,memo] check before implementing step4. do i need both topic_event and topic_lineage?YES and even more...  
[x] - [chat] report generation design doc.  
[x] - [reader] add scriptes to simulate past 12 months data to find the "merge/create" threshold.  
[x] - [reader,tui] ask user to choose a prefered report style:
       Quick background (5–10 min)
       Research briefing (decision-oriented)
       Brainstorm directions (novelty hunting)
       Implementation angle (what to build / how to test)    
[x] - [memo] refactor topic related db tables for report generation  
[x] - [memo,reader] add new memo cmd to record report generation job status in db and integrate with reader.  
[ ] - [reader,memo] fetch the existing topics from db and compute the sim with chosen cluster, and gate to decide create/merge.  
[ ] - [reader,memo] generate the report planner prompt for llm call 1, memo supply metadat if necessary.  
[ ] - [reader,memo] generate the report writter prompt from llm call 2, memo supply metadat if necessary.  
[ ] - [reader,memo] write the whole metadata from reader to memo db, and dump report to local fs.  


random(spikes, explore)  
[ ] - [reader] use lite llm and lite metadata(name, title, keywords) for summarization and thinking llm and depth-aware metadata for report genereation.  
