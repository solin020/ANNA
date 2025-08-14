from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from starlette.config import Config
import numpy as np
import asyncio, aiohttp, aiofiles, os
from functools import cached_property
import numpy as np, base64
from .conversation_controller import ConversationController
import random
from datetime import datetime, timedelta, timezone
from .database import engine, CallLog, Participant, ScheduledCall
from sqlmodel import Session, select
from sqlalchemy.orm.exc import NoResultFound
from ..config import public_url, wss_url, llm_url, grading_url, syntax_url,\
    account_sid, auth_token,\
     word_recordings_directory, call_recordings_directory, sounds_directory
from dataclasses import dataclass
from typing import Optional
from tempfile import NamedTemporaryFile
import uuid
from ..llm import exllama_interact

client = Client(account_sid, auth_token)
STOPWORD_LIST =  ["yes", "sure", "yep", "yeah", "go", "ahead", "next", "ready", "ok", "okay", "continue", "going"]
NEGATIVE_STOPWORD_LIST =  ["no", "nope", "not", "isn't", "don't", "nuh", "aren't", "aint", "wait", "yet", "bit"]

MEMORY_WORDS = os.listdir(word_recordings_directory)
topics = ["your favorite story",
"your favorite trip",
"your favorite food",
"your family",
"your friends",
"the most interesting thing that's ever happened to you",
"your favorite movie",
"your hometown",
"your favorite hobby",
"your favorite game",
"the last dream you can remember",
"your favorite childhood memory",
"what you love most",
"the weather today",
"how you manage stress",]




class PhoneConversationController(ConversationController):
    def convert_to_16khz(self, bytes_):
        b = mulaw_decode_array[
                np.frombuffer(
                    bytes_, dtype='u1'
                )
            ].tobytes()
        blen = len(b)
        #the below line interpolates short 0s into the bytes and then gets the fourier transform of that
        ft = np.fft.fft(np.frombuffer((np.frombuffer(b, '<i2').astype('<i4') << 16).tobytes(), '<i2'))
        #this clears the high frequencies of the fourier tranform
        ft[blen // 2 - blen // 4 : blen // 2 + blen // 4] = 0
        #inverse fft returns the upsampled signal
        return np.fft.ifft(ft).astype('<i2').tobytes()

    @cached_property
    def ffmpeg_convert_to_outbound(self):
        return f"ffmpeg -i {self.temp_file} -ar 8000 -f mulaw -acodec pcm_mulaw {self.output_file}"
    

    @cached_property
    def INBOUND_SAMPLE_RATE(self):
        return 8000

    @cached_property
    def INBOUND_BYTE_WIDTH(self):
        return 1

    @cached_property
    def OUTBOUND_SAMPLE_RATE(self):
        return 8000

    @cached_property
    def OUTBOUND_BYTE_WIDTH(self):
        return 1

    @cached_property
    def OUTBOUND_ZERO_BYTE(self):
        return b'\x7f'







@dataclass
class CallState:
    call_sid: str
    phone_number: str
    call_log: CallLog
    controller: ConversationController
    grading_tasks: list[asyncio.Task]
    previous_calls: int
    uuid: str

    def __init__(self, call_sid: str, phone_number: str, previous_rejects: int):
        self.call_sid = call_sid
        self.phone_number = phone_number
        self.previous_calls = 0
        try:
            with Session(engine) as s:
                statement = select(Participant).where(Participant.phone_number == phone_number)
                result = s.exec(statement).one()
                self.participant_study_id = result.participant_study_id
                statement = select(CallLog).where(CallLog.phone_number == phone_number and CallLog.rejected=="completed")
                result = list(s.exec(statement).all())
                self.previous_calls = len(result)
        except:
            self.participant_study_id = "unknown"
        self.call_log = CallLog(
            call_sid=self.call_sid, 
            phone_number=self.phone_number,
            participant_study_id=self.participant_study_id,
            timestamp = datetime.now(),
            history=[],
            previous_rejects=previous_rejects,
        )
        self.controller = PhoneConversationController()
        self.grading_tasks = []
        self.end_event = asyncio.Event()
        self.uuid = uuid.uuid4().hex

    async def try_end(self):
        self.end_event.set()

    async def time_end(self):
        await asyncio.sleep(900)
        self.end_event.set()

    async def after_call(self, script):
        timer =asyncio.ensure_future(self.time_end())
        script = asyncio.ensure_future(script)
        await self.end_event.wait()
        script.cancel()
        timer.cancel()
        client.calls(self.call_sid).update(status='completed')
        print(f'call {self.call_sid} completed')
        await exllama_interact.delete_session(self.uuid)
        #Cancel any scheduled calls within one hour of complete call
        for task in self.grading_tasks:
            try:
                await task
            except Exception as e:
                print(e)
        outbound_pcm_file = NamedTemporaryFile(suffix='.pcm', delete=False).name
        internal_pcm_file = NamedTemporaryFile(suffix='.pcm', delete=False).name
        async with aiofiles.open(internal_pcm_file, 'wb+') as f:
            await f.write(self.controller.participant_track)
        async with aiofiles.open(outbound_pcm_file, 'wb+') as f:
            await f.write(self.controller.outbound_bytes)
        proc = await asyncio.create_subprocess_shell(
            (f'ffmpeg -ar 8k -f mulaw -i {outbound_pcm_file} -ar 16k -f s16le -i {internal_pcm_file}'
            ' -filter_complex "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]" -map "[a]" ')
            + os.path.join(call_recordings_directory, f'call_{self.call_sid}.wav') + 
            (f'&& rm {internal_pcm_file} {outbound_pcm_file}'),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print(await proc.communicate(), flush=True)
        if self.call_log.previous_rejects < 1 and self.call_log.rejected !='completed':
            await self.reschedule_call(self.call_log.previous_rejects+1)
        with Session(engine) as s:
            s.add(self.call_log)
            s.commit()

    async def handle_streams(self, frame, websocket, stream_sid):
        await self.controller.receive_inbound(base64.b64decode(frame['media']['payload']))
        inbytes = self.controller.get_speech_bytes()
        if inbytes:
            media_data = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                   "payload": base64.b64encode(inbytes).decode('utf-8')
                }
            }
            await websocket.send_json(media_data)

    @staticmethod
    def begin_conversation():
        vr = VoiceResponse()
        connect = Connect()
        connect.stream(url=f'{wss_url}/stream-conversation-socket')
        vr.append(connect)
        return vr




    @staticmethod
    async def outbound_call(phone_number: str, previous_rejects:int):
        call = client.calls.create(
            to='='+phone_number,
            from_='+16126826292',
            twiml = CallState.begin_conversation().__str__(),
        )
        if call.sid:
            self = CallState(phone_number=phone_number,call_sid=call.sid,previous_rejects=previous_rejects)
            script = self.participant_initiated_script()
            asyncio.ensure_future(self.after_call(script))
            return self
        else:
            raise Exception("Twilio failure")

    @staticmethod
    async def inbound_call(form):
        call_sid = form['CallSid']
        phone_number = form['From']
        self = CallState(call_sid=call_sid, phone_number=phone_number, previous_rejects=999)
        script = self.bot_initiated_script()
        asyncio.ensure_future(self.after_call(script))
        with Session(engine) as s:
            statement = select(ScheduledCall).where(
                ScheduledCall.phone_number == phone_number
            )
            results = list(s.exec(statement))
            for r in results:
                if (r.time - datetime.now()) < timedelta(hours=1) and r.time > datetime.now():
                    print(f'cancelling next call {r.id}')
                    try:
                        os.system(f'at -r {r.id}')
                    except Exception as e:
                        print(e)
                    s.delete(r)
            s.commit()
        return self
    
    async def say(self, quote:str="", file:str="", **kwargs):
        self.call_log.history.append(("SYSTEM", quote))
        await self.controller.say(quote=quote, file=file, history=self.call_log.history, **kwargs)

    async def ask(self, quote, file:str="", forward_to_llm=False, **kwargs):
        self.call_log.history.append(("SYSTEM", quote))
        if forward_to_llm:
            await exllama_interact.bot_say(quote)
        print('got to ask')
        reply =  await self.controller.ask(question=quote, history=self.call_log.history, file=file, **kwargs)
        self.call_log.history.append(("USER", reply))
        return reply
        
    
    async def perplexity_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{llm_url}/perplexity', json=self.call_log.history) as resp:
                self.call_log.perplexity_grade = await resp.json()
                print('perplexity', self.call_log.perplexity_grade)
    
    async def syntax_grade(self):
            async with aiohttp.ClientSession() as session:
               syntax_grade = []
               for speaker, sentence in self.call_log.history:
                   if speaker == "USER":
                       async with session.post(f'{syntax_url}/', json=sentence) as resp:
                           sg = await resp.json()
                           syntax_grade.append(sg)
            self.call_log.syntax_grade = syntax_grade
            print("syntax grade", self.call_log.syntax_grade)

    async def memory_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-memory-test', json={
                    'transcript': self.call_log.memory_exercise_reply,
                    'word_list': self.call_log.memory_exercise_words
                }) as resp:
                    print('reply 1', self.call_log.memory_exercise_reply)
                    print('words', self.call_log.memory_exercise_words)
                    self.call_log.memory_grade = await resp.json()
                    print("memory grade", self.memory_grade)
    
    async def l_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-l-test', json=self.call_log.l_reply) as resp:
                self.call_log.l_grade = await resp.json()
    
    async def f_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-f-test', json=self.call_log.f_reply) as resp:
                self.call_log.f_grade = await resp.json()
    
    async def c_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-c-test', json=self.call_log.c_reply) as resp:
                self.call_log.c_grade = await resp.json()

    async def animal_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-animal-test', json=self.call_log.animal_reply) as resp:
                self.call_log.animal_grade = await resp.json()
    
    async def vegetable_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-vegetable-test', json=self.call_log.vegetable_reply) as resp:
                self.call_log.vegetable_grade = await resp.json()
    
    async def fruit_grade(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-fruit-test', json=self.call_log.fruit_reply) as resp:
                self.call_log.fruit_grade = await resp.json()

    async def memory_grade_2(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(f'{grading_url}/grade-memory-test', json={
                    'transcript': self.call_log.memory_exercise_reply_2,
                    'word_list': self.call_log.memory_exercise_words
                }) as resp:
                self.call_log.memory_grade_2 = await resp.json()
                print('reply 2', self.call_log.memory_exercise_reply_2)
                print('words', self.call_log.memory_exercise_words)
                print('memory grade 2', self.call_log.memory_grade_2)
    
    async def reschedule_call(self, rejects:int):
        new_time = datetime.now() + timedelta(minutes=10)
        from . import app
        t = app.ScheduledCall(id=self.phone_number+new_time.isoformat(), 
                            time=new_time, 
                            phone_number=self.phone_number,
                            rejects=rejects)
        await app.scheduled_call_task(t.time, t.phone_number, rejects)





 #    
        
    #TODO: automate addition to history.
    async def bot_initiated_script(self):
        print('got to bot initiated script', flush=True)
        if self.previous_calls > 4:
            await self.short_initiated_script()
            return
        print('began talking')
        #this parameter is set by default until the participant gets through the script
        self.call_log.rejected = "ignored"
        await self.say("Hello! My name is Anna. I'm your automated nursing assistant.", start_label="long_intro")
        await self.say("My job is to engage in conversation with you and help to detect any temporary changes you may develop in your speech or memory as a result of car tea treatment.")
        await self.say("This call will be recorded only for research purposes and analyzed by research staff. Nothing you say will be judged or compared to others.")
        await self.say("Because I'm a robot, I might sometimes take a little longer to respond, or I might say something that doesn’t quite make sense. If that happens, please be patient and stay on the line.")
        await self.say("You can tell me to continue at any time by saying the word continue.")
        await self.say("It’s best if you can find a quiet place for our chat, but it's okay if there’s some noise around you. I’ll listen for pauses in your speech to know when it’s my turn to talk.")
        await self.say("In the first part of this call, I will check for changes in how you speak caused by the medication you are receiving.")
        await self.say("I will suggest a topic and just need you to relax and converse for as long as you are comfortable doing.")
        print('finished intro')
        #this prepares the llm for the conversation
        await exllama_interact.setup_session(self.uuid)
        await self.ask_permission("Are you ready to have a chat?", end_label="long_intro")
        self.call_log.rejected = "accepted"
        #free conversation
        init_question = ("Okay. First, tell me how you've been doing in the last few hours?")
        await self.ask(init_question, 
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                await_silence=True,
                                silence_window=2, stopword_list=['continue'],
                                start_label="how_are_you_doing", end_label="how_are_you_doing"
                                )
        await self.ask("Thank you for sharing. Is there anything else you want to share about your day so far?",  start_label="how_are_you_doing_2", end_label="how_are_you_doing_2",
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                await_silence=True,
                                silence_window=2, stopword_list=['continue'])
        topic = topics[self.previous_calls % len(topics)]
        topic_question = (f"Thank you for sharing. Now, I would like to ask you to tell me as much as you are able about {topic}. "
                                "Feel free to take your time in answering.")
        await self.ask(topic_question,
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                await_silence=True,
                                silence_window=2,
                                stopword_list=['continue'],
                                start_label="topic_question", end_label="topic_question",
                                wait_time=30)
        await self.ask(await self.get_response(), await_silence=True, minimum_turn_time=10, stopword_list=['continue'], wait_time=30,  start_label="topic_question_2", end_label="topic_question_2",)
        await self.ask(await self.get_response(), await_silence=True, minimum_turn_time=10, stopword_list=['continue'], wait_time=30,  start_label="topic_question_3", end_label="topic_question_3",)
        self.grading_tasks.append(asyncio.ensure_future(self.perplexity_grade()))
        self.grading_tasks.append(asyncio.ensure_future(self.syntax_grade()))

        #word list recall task
        await self.say("Thank you. Now let's move on to the next part of the call in which I will ask you to complete several cognitive tasks to check for any medication related changes in your short term memory and attention.",
                       end_label="memory_permission")
        await self.say("There are no right or wrong responses here. Just try to complete these tasks as best you can.")
        await self.say("First, I am going to read aloud six words. Please repeat each word you hear aloud. Later, I will ask you to recall all six words.")
        await self.ask_permission("Are you ready?", 
                        end_label="memory_description")
        await self.say("Here is the list.", start_label="memory_begin", end_label="memory_begin")
        word_files = random.sample(MEMORY_WORDS, k=6)
        self.call_log.memory_exercise_words = [w.split('.')[0] for w in word_files]
        for w in word_files:
            await self.say(file=os.path.join(word_recordings_directory,w), final_pause=1.0, initial_pause=1.0, start_label="memory_word", end_label="memory_word")
        self.call_log.memory_exercise_reply = await self.ask("Now repeat as many of these words as you remember and say continue when you are done. Please begin after the beep.",
                               file=os.path.join(sounds_directory,"beep.wav"), wait_time=30,  stopword_list=['continue'],
                               start_label="memory_response", end_label="memory_response")
        self.grading_tasks.append(asyncio.ensure_future(self.memory_grade()))

        #L initial word list task
        await self.say("Thank you. Now, I will give you three letters of the alphabet one at a time.", start_label="l_permission")
        await self.say("I am going to ask you to name words that begin with that letter, as fast as you are able.")
        await self.say("For example, if I give you the letter ess, as in sam, you may say soft, smile, and so on.")
        await self.say("Please don't use the same word with different endings, like smiling, smiled, and smiles.")
        await self.ask_permission("Are you ready?", end_label="l_permission")
        await self.say("Okay. Your letter is the letter ell, as in laugh, or ladle.", start_label="l_begin")
        await self.say("Please name all the words that you are able to think of that begin with the letter ell.")
        self.call_log.l_reply = await self.ask("You have thirty seconds. Please begin after the beep.",  
            end_label="l_begin",
            file=os.path.join(sounds_directory,"beep.wav"), 
            wait_time=30, 
            stopword_list=['continue'])
        self.grading_tasks.append(asyncio.ensure_future(self.l_grade()))

        #C initial word list task
        await self.say("Please stop. Thank you.", start_label="c_permission")
        await self.ask_permission("Are you ready for the next letter?", end_label="c_permission")
        await self.say("Okay. Your letter is the letter see, as in coffee, or cinema.", start_label="c_begin")
        await self.say("Please name all the words that you are able to think of that begin with the letter see.")
        self.call_log.c_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
            end_label="c_begin",
            file=os.path.join(sounds_directory,"beep.wav"), 
            wait_time=30, 
            stopword_list=['continue'])
        self.grading_tasks.append(asyncio.ensure_future(self.c_grade()))

        #F initial word list task
        await self.say("Please stop. Thank you.", start_label="f_permission")
        await self.ask_permission("Are you ready for the last letter?", end_label="f_permission")
        await self.say("Okay. Your letter is the letter F, as in finger, or floor.", start_label="f_begin")
        await self.say("Please name all the words that you are able to think of that begin with the letter F.")
        self.call_log.f_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
            end_label="f_begin",
            file=os.path.join(sounds_directory,"beep.wav"), 
            wait_time=30, 
            stopword_list=['continue'])
        self.grading_tasks.append(asyncio.ensure_future(self.f_grade()))


        #animal naming word list task
        await self.say("Thank you. Now, I will give you three categories one at a time.", start_label="animal_permission")
        await self.say("I am going to ask you to name as fast as you are able all the things that belong to that category.")
        await self.say("For example, if I give you the category of articles of clothing, you may say shirt, or jacket, or pants, and so on.")
        await self.ask_permission("Are you ready?", end_label="animal_permission")
        await self.say("Okay. Your category is animals.", start_label="animal_begin") 
        await self.say("Name as many animals as you are able to think of.")
        self.call_log.animal_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
            end_label="animal_begin",
            file=os.path.join(sounds_directory,"beep.wav"), 
            wait_time=30, 
            stopword_list=['continue'])
        self.grading_tasks.append(asyncio.ensure_future(self.animal_grade()))

        #fruit naming word list task
        await self.say("Please stop. Thank you.", start_label="fruit_permission")
        await self.ask_permission("Are you ready for the next category?", end_label="fruit_permission")
        await self.say("Okay. Your category is fruits.", start_label="fruit_begin") 
        await self.say("Name as many fruits as you are able to think of.")
        self.call_log.fruit_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
            end_label="fruit_begin",             
            file=os.path.join(sounds_directory,"beep.wav"), 
            wait_time=30, 
            stopword_list=['continue'])
        self.grading_tasks.append(asyncio.ensure_future(self.fruit_grade()))

        #vegetable naming word list task
        await self.say("Please stop. Thank you.", start_label="vegetable_permission")
        await self.ask_permission("Are you ready for the last category?", end_label="vegetable_permission")
        await self.say("Okay. Your category is vegetables.", start_label="vegetable_begin") 
        await self.say("Name as many vegetables as you are able to think of.")
        self.call_log.vegetable_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
            end_label="vegetable_begin",
            file=os.path.join(sounds_directory,"beep.wav"), 
            wait_time=30, 
            stopword_list=['continue'])
        self.grading_tasks.append(asyncio.ensure_future(self.vegetable_grade()))

        #goodbye
        await self.say("Okay. Please stop. This concludes our session.", start_label="goodbye")
        await self.say("Thank you for your patience in completing this longer baseline call.")
        await self.say("The future calls will be shorter. I will give you only one of the three letters and only one of the three categories in future calls.")
        self.call_log.rejected = "completed"
        await self.ask("Until next time. Goodbye.", wait_time=0.5, end_label="goodbye")
        await self.try_end()


    def participant_initiated_script(self):
        return self.bot_initiated_script()

    async def short_initiated_script(self):
        print('began short script')
        #this parameter is set by default until the participant gets through the script
        self.call_log.rejected = "ignored"
        await self.say("Hello, this is Anna - your automated nursing assistant. This call will be recorded for research use.", start_label="short_intro")
        await self.say("As in previous calls, first we chat about a random topic.")
        await self.say("This call will be recorded only for research purposes and analyzed by research staff. Nothing you say will be judged or compared to others.")
        await self.say("Remember - the goal is simply to talk for as long as you are comfortable doing.")
        print('finished intro')
        #this prepares the llm for the conversation
        await exllama_interact.setup_session(self.uuid)
        await self.ask_permission("Are you ready to have a chat?", end_label="short_intro")
        self.call_log.rejected = "accepted"
        #free conversation
        init_question = ("Okay. First, tell me how you've been doing in the last few hours?")
        await self.ask(init_question, 
                                start_label="how_are_you_doing",
                                end_label="how_are_you_doing",
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                await_silence=True,
                                silence_window=2, stopword_list=['continue'])
        await self.ask("Thank you for sharing. Is there anything else you want to share about your day so far?", 
                                start_label="how_are_you_doing_2",
                                end_label="how_are_you_doing_2",
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                await_silence=True,
                                silence_window=2, stopword_list=['continue'])
        topic = topics[self.previous_calls % len(topics)]
        letter = ['l', 'c', 'f'][self.previous_calls % 3]
        category = ['animal', 'fruit', 'vegetable'][self.previous_calls % 3]
        topic_question = (f"Thank you for sharing. Now, I would like to ask you to tell me as much as you are able about {topic}. "
                                "Feel free to take your time in answering.")
        await self.ask(topic_question,
                                start_label="topic_question",
                                end_label="topic_question",
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                await_silence=True,
                                silence_window=2,
                                stopword_list=['continue'],
                                wait_time=30)
        await self.ask(await self.get_response(), await_silence=True, minimum_turn_time=10, stopword_list=['continue'], wait_time=30,
                                start_label="topic_question_2",
                                end_label="topic_question_2",)
        await self.ask(await self.get_response(), await_silence=True, minimum_turn_time=10, stopword_list=['continue'], wait_time=30,
                                start_label="topic_question_3",
                                end_label="topic_question_3",)
        self.grading_tasks.append(asyncio.ensure_future(self.perplexity_grade()))
        self.grading_tasks.append(asyncio.ensure_future(self.syntax_grade()))

        #word list recall task
        await self.say("Thank you. Now, on to the cognitive tasks. Remember, no right or wrong responses here.",
                       start_label="memory_permission")
        await self.say("First, I am going to ask you to remember a list of six words.")
        await self.ask_permission("Are you ready?", end_label="memory_permission")
        await self.say("Here is the list.", start_label="memory_begin", end_label="memory_begin")
        word_files = random.sample(MEMORY_WORDS, k=6)
        self.call_log.memory_exercise_words = [w.split('.')[0] for w in word_files]
        for w in word_files:
            print('got to word', w, flush=True)
            await self.say(file=os.path.join(word_recordings_directory,w), final_pause=1.0, initial_pause=1.0, start_label="memory_word", end_label="memory_word")
        self.call_log.memory_exercise_reply = await self.ask("Please begin after the beep and say continue when you are done.",
                                start_label="memory_response", end_label="memory_response",
                               file=os.path.join(sounds_directory,"beep.wav"), wait_time=30,  stopword_list=['continue'])
        self.grading_tasks.append(asyncio.ensure_future(self.memory_grade()))
        await self.say("Thank you. Now, I will give you a letter of the alphabet.", start_label=f"{letter}_permission")
        await self.ask_permission("Are you ready?", end_label=f"{letter}_permission")
        if letter == "l":
            #L initial word list task
            await self.say("Okay. Your letter is the letter ell, as in love.", start_label=f"{letter}_begin")
            await self.say("Please name all the words that you are able to think of that begin with the letter ell.")
            self.call_log.l_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
                end_label=f"{letter}_begin",
                file=os.path.join(sounds_directory,"beep.wav"), 
                wait_time=30, 
                stopword_list=['continue'])
            self.grading_tasks.append(asyncio.ensure_future(self.l_grade()))
        elif letter == "c":
        #C initial word list task
            await self.say("Okay. Your letter is the letter see, as in cat.", start_label=f"{letter}_begin")
            await self.say("Please name all the words that you are able to think of that begin with the letter see.")
            self.call_log.c_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
                end_label=f"{letter}_begin",
                file=os.path.join(sounds_directory,"beep.wav"), 
                wait_time=30, 
                stopword_list=['continue'])
            self.grading_tasks.append(asyncio.ensure_future(self.c_grade()))
        elif letter == "f":
            #F initial word list task
            await self.say("Okay. Your letter is the letter F, as in finger.", start_label=f"{letter}_begin")
            await self.say("Please name all the words that you are able to think of that begin with the letter F.")
            self.call_log.f_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
                end_label=f"{letter}_begin",
                file=os.path.join(sounds_directory,"beep.wav"), 
                wait_time=30, 
                stopword_list=['continue'])
            self.grading_tasks.append(asyncio.ensure_future(self.f_grade()))
        await self.say("Thank you. Now, I will give you a category. ", start_label=f"{category}_permission")
        await self.ask_permission("Are you ready?", end_label=f"{category}_permission")
        if category=="animal":
            #animal naming word list task
            await self.say("Okay. Your category is animals.", start_label=f"{category}_begin") 
            self.call_log.animal_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
                end_label=f"{category}_begin",
                file=os.path.join(sounds_directory,"beep.wav"), 
                wait_time=30, 
                stopword_list=['continue'])
            self.grading_tasks.append(asyncio.ensure_future(self.animal_grade()))
        elif category=="fruit":
            #fruit naming word list task
            await self.say("Okay. Your category is fruits.", start_label=f"{category}_begin") 
            self.call_log.fruit_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
                end_label=f"{category}_begin",
                file=os.path.join(sounds_directory,"beep.wav"), 
                wait_time=30, 
                stopword_list=['continue'])
            self.grading_tasks.append(asyncio.ensure_future(self.fruit_grade()))
        elif category=="vegetable":
            #vegetable naming word list task
            await self.say("Okay. Your category is vegetables.", start_label=f"{category}_begin") 
            self.call_log.vegetable_reply = await self.ask("You have thirty seconds. Please begin after the beep.", 
                end_label=f"{category}_begin",
                file=os.path.join(sounds_directory,"beep.wav"), 
                wait_time=30, 
                stopword_list=['continue'])
            self.grading_tasks.append(asyncio.ensure_future(self.vegetable_grade()))

        #goodbye
        await self.say("Thank you. This concludes our session.", start_label="goodbye")
        self.call_log.rejected = "completed"
        await self.ask("Until next time. Goodbye.", wait_time=0.5, end_label="goodbye")
        await self.try_end()

    async def ask_permission(self, quote, start_label="", end_label=""):
        current_hour = datetime.now().hour
        call_permission = await self.ask(quote=quote, start_label=start_label, end_label=end_label, stopword_list=STOPWORD_LIST+NEGATIVE_STOPWORD_LIST, wait_time=30, return_stopword=True)
        print(f'{call_permission=}')
        if any(n in call_permission.lower() for n in NEGATIVE_STOPWORD_LIST) or not any(y in call_permission.lower() for y in STOPWORD_LIST):
            call_permission_2 = await self.ask(quote="Okay. Say I'm ready when you're ready.", stopword_list=['ready'], wait_time=60)
            rejects = self.call_log.previous_rejects+1
            if not any(y in call_permission_2.lower() for y in STOPWORD_LIST):
                print('call again later')
                if rejects < 2:
                    await self.ask(quote="Okay. I will call you again in ten minutes.", wait_time=1, end_label="call_again")
                else:
                    if 0<=current_hour<12:
                        await self.ask(quote="Okay. I will call you again in the afternoon.", wait_time=1, end_label="call_again")
                    elif 12<=current_hour<17:
                        await self.ask(quote="Okay. I will call you again in the evening.", wait_time=1, end_label="call_again")
                    else:
                        await self.ask(quote="Okay. I will call you again tomorrow morning.", wait_time=1, end_label="call_again")
                client.calls(self.call_sid).update(status='completed')


    async def get_response(self):
        filtered_history = [h for h in self.call_log.history if h[0] == 'USER' or h[0] == 'SYSTEM']
        async with aiohttp.ClientSession() as session:
            async with session.post(f'{llm_url}/generate', json=filtered_history) as resp:
                bot_says = await resp.text()
        print('bot', bot_says, flush=True)
        return bot_says
   

class EndCall(Exception):
    pass

   


        
mulaw_decode_array = np.array(
      [-32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
       -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
       -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
       -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
        -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
        -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
        -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
        -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
        -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
        -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
         -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
         -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
         -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
         -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
         -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
          -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
        32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
        23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
        15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
        11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
         7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
         5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
         3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
         2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
         1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
         1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
          876,    844,    812,    780,    748,    716,    684,    652,
          620,    588,    556,    524,    492,    460,    428,    396,
          372,    356,    340,    324,    308,    292,    276,    260,
          244,    228,    212,    196,    180,    164,    148,    132,
          120,    112,    104,     96,     88,     80,     72,     64,
           56,     48,     40,     32,     24,     16,      8,      0] 
, dtype='<i2')
