main(feature level)  
[ ] - discuss memory cli log with chat, --debug  
[ ] - sql performance log maybe  
[ ] - add e2e test for reader and memo  
[ ] - algo version control: move algo_libs to a new git repo  
[ ] - [cursor] set minimum functional requirements to force coding agent follow them while refactoring.  
[ ] - [eval] implement a llm request&response cache mechanism.comment.need more clear use case for what to cahe/not cache and why.  
[ ] - [eval, reader] design a structure to share the llm evaluation logic between eval pipeline and reader. f.e use the llm heuristic rules and judge from reader.  
[ ] - [memo] enforce machine-only design in logging system: diagnostic logs -> stderr, cmd output -> stdout  



[x] - [memo,agent-friendly] add sample use case for each subcmd in their help message.  
[ ] - [reader] add paper metadata extraction to hf metadata processing step(fetch+embed+clustering) and also adjust the evidence provieded in report planner's specs.    
[...] - [lab] consider make lab an independent repo so we can have version control over all its ingredients via git commit hash.  
[ ] - [reader] make "selectors" used in call1/call2 load/generate dynamically from memo, this should be a prod-level feature. comment. the design is selectors in reader are semantic ones(agent-facing) and the selector in dbs are db fields. the current implementation is memo holds a simple one to one map(case insensitive exact match), we may consider lexical + embedding(retrival+ranking) when the relationship between these two layers cannot be captured fully by case insensitive exact match. or the agent interface requires a dedicate db search memo cmd.  
[ ] - [?] a vibe-coding like structure to derive pydantic model from prompt spec, so no need to validate these two using validate.py. "spec" holds the high-level design, "prompt"(semantic meanings) and "structure"(code model) should be derived from it. reason for not having it and must have validate - when reader moves away from python to rust, and there is no pydantic-like rust?    
[ ] - [reader] add new pipe/workflow for re-run missing cluster observation due to llm call 429 RESOURCE_EXHAUSTED error. comment. a code agent workflow using rerun helper in log.  
[ ] - [memo] async version wrt db query.    

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
[x] - [reader,memo] add a new algo-lib topic-resolver to fetch the existing topics from db and compute the sim with chosen cluster, and gate to decide create/merge.  
[x] - [reader,memo] generate the report planner prompt for llm call 1, memo supply metadat if necessary.  
[x] - [reader] benchmark pdf libs for paper introduction extraction.  
[x] - [reader] add prompt spec and builder to for llm report planner prompt generation under prompts/report_planner.  
[x] - [reader,memo] add new memo cmd get-report-planner-metadata for call 1 planner prompt generation and integrate with reader.  
[x] - [memo] add design docs for memo wrt how to store and query paper metadata for call1 and call2.  
[x] - [lab] draft a lab env for better test environment.  
[x] - [reader] refine call 1 prompt spec baseline version for better call1&call2 evidence composition.  
[x] - [chat] Q: it's easy for workflow to ensure the "providing evidence a, b, c", but it's hard for workflow to evaluate the quality of "evidence a, b, c" to derive a solid decision, and the purpose of call 1 prompt spec baseline is to iterate on the quality of the evidence to hopefully derive a solid plan with fewer iterations. and yes there are some if-else rules set in prompt spec, but they are also guidances/hypethesis(not constrains) for model to derive a solid plan.    
[x] - [reader] refactor call1 prompt and output pydantic models to collect llm instructions for evidence/memo metadata collection.  
[x] - [paper-chunk] prepare env(especially paper chunk pipeline) for call1 test.note setup using paper_chunk.    
[x] - [reader] create separated folder for hf data pipeline. note. pipelines/hf_data  
[x] - [paper-chunk] run phase 1 script to get phase 1 heristic rules and also phase 2 candidates.  
[x] - [paper-chunk] refactor paper curation script to aysnc+multi-thread, refactor alias regex compiler for multi-phrase match.  
[?] - [paper-chunk] phase 2 design & implementation. puspend. waiting for call1&call 2 feddback.        
[x] - [paper-chunk] finalized the rules.yaml in phase 1 and fix unmapped header suggest candidate confidence normalization and cross paper voting issues.  
[x] - [reader] add new paper-chunk algo-lib.  
[x] - [reader] integrate paper-chunk algo-lib into hf_data pipe.  
[x] - [memo] refactor fresh-paper cmd to return paper cluster details by default.  
[x] - [memo,reader] new memo cmd to write paper-chunk data/versioning to db. note. new cmd inject-papers-chunk added.      
[x] - [memo,reader] integration test paperchunk algo-lib, memo inject-papers-chunk, reader hf_data.  
[x] - [reader] port cluster summarization step from monthly pipe to hf_data.  
[x] - [reader] refactor hf_data pipe to async.  

[x] - [reader] run hf_data pipe to prepare data for call 1.  
[x] - [reader] refine hf_data for better error shooting and multi-purpose configuratons and add retry for overall_score is 0's llm summarization calls.  
[x] - [reader] attach suggestions based on judge valid errors in llm prompt during retry.  

[x] - [reader] test call 1 llm report planner(heuristic rules for response?).  
[x] - [reader] refactor chosen topic + report generation to an independent async workflow(report-generation) under pipelines.  
[x] - [reader] add heuristic rules for call 1 and call 1 retry based on heuristic rules overall score  
[x] - [reader] create call 1 planner production spec, only work on plan evidence completion, remove next step input.  
[x] - [reader] refactor the design for report generation into call 1 report plan(with evidence collection iterations) + call 1.5 (report writting evidence collection based on plan's outline and other fields) + call 2 (pure writing with outline and call 1.5 evidence). 
[x] - [reader] implement call 1 evidence collection stop line design.  
[x] - [reader,memo] rework get report planner evidence cmd to collect summary-level call 1 evidance as a must-haev for each call.  
[x] - [reader,memo] add new memo cmd to collect supplement evidence for the call 1 evidence gaps requests.  
[x] - [reader] refactor the llm call judge loop, evidence collection loop and call 1 step with detailed status.  
[x] - [reader] redesign report generation to step 1 write report(call 1) + step 2 write by outline: loop plan.outline(call 2n one section evidence fetch + call 3n one section writing) + step 3 write report name, summary, keywords(call 4)  
[x] - [reader] attach new report class for step 2 and 3 and also their prompt.  
[x] - [reader,memo] remove redundent geometric fields from memo get report metadata cmd.  
[x] - [reader] refactor judge_output to a judge class using different metrics as this workflow will be shared by all the steps.  
[x] - [reader] port planner's llm call retry based on judge result to a generic method of LLMClient as this function will be shared by all the steps.  
[x] - [reader] implement step 2.  
[x] - [reader] implement step 3.    
[x] - [reader] design and implement a workflow register to cache each step's artifact and status to local fs(cache) for error runtime retry.  
[x] - [reader,memo] design the report generation db update workflow.note. resolve the one cluster event has more than one reports, or report regeneration issue. [comment] once a cluster is consumed(tights to one report in db), it's done, no report regeneration for it.  
[x] - [reader] design resume failure job from cache. comment. this will be handled by agent, out of the report generation workflow, the workflow only keeps the workflow register to save traces and outputs and return resume help message.  
[x] - [reader,memo] move report_job from memo db to a reader owned sqlite db, the report job tracking is now controled by the report generation pipe.  
[x] - [reader] finalize the report_generation pipe by wrapping it into a step function which return next action and msg.  

[x] - [reader] test report generation in one.  
[x] - [reader] fix logging display issue(disable markup)  
[x] - [reader] organize logging in report generation pipeline  
[x] - [reader] add user intent in workflow cache json  
[x] - [memo] add new cmds get-report and check-report-signature  
[x] - [reader] add new pip render-report for fetch+check signature+tui view the generated report.  

[x] - [reader] test new pip render-report
[x] - [reader] split the render-report cmd into 3 cmds between 2 tools: memo get-report + reader check-report-signature + reader render-report.
[ ] - [doc] add readme for whole repo  

[ ] - [ci/cd] add agent job to dymically update RULES.md and README like doc before each push.  

[ ] - [reader] provide an agent interface.  


[ ] - [reader] **consider kmeans++?**  
[ ] - [memo] auto db migration. because the sqlite is designed to be a local db/memory system and is tighted to the memo cli. woould be interested to think of a way to handle local db migration after cli upgrade. a classical way just use a bunch of migration scripts. but since this is a machine-only/agent facing tool, maybe the migration can be some sort of prompt files and let agent drives/suggests the local db change vs new db schema.  
[ ] - [reader] now we have tokenbucket per pipeline execution time. consider adding a second layer of tokenbucket for per llm provider: pipeline token bucket -> llm provider token bucket. and also consider caching them in db.  
[ ] - [memo,agent-friendly] refactor the error handling to a descriptive way, f.e. add error type for each cmd and raise them with more details of cmd runtime.  
 

[ ] - [paper-chunk] test training mode and remove duplicate code under paper_chunk dir.  
[ ] - [paperchunk] refactor pdf extractor, it is in poor condition..note. pdf extractor is disbaled now, only html is enabled.      
[ ] - [reader] draft code agent integration for auto-scheduled rerun wrt rerun helper.  
[ ] - [hf-data] collect pipeline events as jsonl for performance analysis.  
[ ] - [report-generation] remove spec_baseline and its deps, f.e. NextStepInput.  


random(spikes, explore)  
[x] - [reader] use lite llm and lite metadata(name, title, keywords) for summarization and thinking llm and depth-aware metadata for report genereation. 