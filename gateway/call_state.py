from twilio.twiml.voice_response import VoiceResponse, Connect, Gather, Hangup
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

nineties = ['ninety one', 'ninety two', 'ninety three', 'ninety four',
            'ninety five', 'ninety six', 'ninety seven', 'ninety eight', 'ninety nine']




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
            miscellaneous = {'script_version': 9},
        )
        self.controller = PhoneConversationController()
        self.end_event = asyncio.Event()
        self.uuid = uuid.uuid4().hex

    async def try_end(self):
        self.end_event.set()

    async def time_end(self):
        await asyncio.sleep(900)
        self.end_event.set()

    async def after_call(self, script):
        from .app import call_dict
        timer =asyncio.ensure_future(self.time_end())
        script = asyncio.ensure_future(script)
        await self.end_event.wait()
        script.cancel()
        timer.cancel()
        client.calls(self.call_sid).update(status='completed')
        print(f'call {self.call_sid} completed')
        await exllama_interact.delete_session(self.uuid)
        #Cancel any scheduled calls within one hour of complete call
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
        call_dict.pop(self.call_sid)

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
            script = self.bot_initiated_script()
            asyncio.ensure_future(self.after_call(script))
            return self
        else:
            raise Exception("Twilio failure")

    @staticmethod
    async def inbound_call(form):
        call_sid = form['CallSid']
        phone_number = form['From']
        self = CallState(call_sid=call_sid, phone_number=phone_number, previous_rejects=999)
        script = self.participant_initiated_script()
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
        await self.controller.say(quote=quote, file=file, history=self.call_log.history, **kwargs)

    async def ask(self, quote, file:str="", forward_to_llm=False, **kwargs):
        if forward_to_llm:
            await exllama_interact.bot_say(quote)
        print('got to ask')
        reply =  await self.controller.ask(question=quote, history=self.call_log.history, file=file, **kwargs)
        self.call_log.history.append(("USER", reply))
        return reply
        
    
    
    async def reschedule_call(self, rejects:int):
        new_time = datetime.now() + timedelta(minutes=10)
        from . import app
        t = app.ScheduledCall(id=self.phone_number+new_time.isoformat(), 
                            time=new_time, 
                            phone_number=self.phone_number,
                            rejects=rejects)
        await app.scheduled_call_task(t.time, t.phone_number, rejects)




    async def bot_initiated_script(self):
        print('got to bot initiated script', flush=True)
        self.call_log.miscellaneous['direction'] = 'outbound'
        if self.previous_calls > 9999:
            await self.say("Hello! My name is Anna. I'm an automated nursing assistant that is part of a research study. This call will be recorded for research use.", start_label="short_intro")
            await self.short_script()
            return
        else:
            await self.say("Hello! My name is Anna. I'm an automated nursing assistant that is part of a research study.", start_label="long_intro")
            await self.long_script()
            return
 #    
        
    #TODO: automate addition to history.
    async def long_script(self):
        print('began talking')
        #this parameter is set by default until the participant gets through the script
        self.call_log.rejected = "ignored"
        await self.say("My job is to engage in conversation with you and help to detect any temporary changes you may develop in your speech or memory as a result of car tea treatment.")
        await self.say("This call will be recorded only for research purposes and analyzed by research staff. Nothing you say will be judged or compared to others.")
        await self.say("Because I'm a robot, I might sometimes take a little longer to respond, or I might say something that doesn't quite make sense. If that happens, please be patient and stay on the line.")
        await self.say("You can tell me to continue at any time by saying the word, continue")
        await self.say("It's best if you can find a quiet place for our chat, but it's okay if there's some noise around you. I'll listen for pauses in your speech to know when it's my turn to talk.")
        await self.say("In the first part of this call, I will check for changes in how you speak caused by the medication you are receiving.")
        await self.say("Please keep your phone close to your mouth if using it as a speakerphone.")
        print('finished intro')
        #this prepares the llm for the conversation
        await self.ask_permission("Are you ready to talk?", end_label="long_intro")
        self.call_log.rejected = "accepted"
        #free conversation
        await self.say('Okay. First, I need to know if there were any new changes in your ability to think, or speak, or in your mood, since our last conversation. ')
        await self.say('Even if these changes seem very minor, please describe them in as much detail as you can.')
        await self.say('Some examples of such changes include but are not limited to the following.')
        await self.say('New headache, or dizziness, or difficulty concentrating, or fuzzy thinking, or hallucinations, or difficulty with finding words when speaking, or sudden mood swings, or new anxiety.')
        init_question = ("If no changes to report, just describe how you have been doing in general in the past few hours. Please begin after the beep.")
        await self.ask(init_question, 
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                await_silence=True,
                                silence_window=2, stopword_list=['continue'],
                                start_label="how_are_you_doing", end_label="how_are_you_doing",
                                file=os.path.join(sounds_directory,"beep.wav")
                                )
        await self.memory_test_loop('long')
        await self.say("Thank you. Now, the math task. You will need to use your phone's keypad to enter your answers using your fingers instead of your voice.", start_label='math_intro')
        await self.say("Please switch your phone to the speakerphone mode by pressing the Speaker button and then bring up the keypad by pressing the Keypad button.")
        first_pound = await self.ask("Press the pound key on your phone's keypad when you are ready. ", wait_time=10, dtmf_wait_character='#')
        #this will happen if no digits were pressed
        if first_pound == True:
           second_pound = await self.ask("Please press the pound key on your phone's keypad when you are ready. ", wait_time=10, dtmf_wait_character='#')
           if second_pound == True:
                current_hour = datetime.now().hour
                rejects = self.call_log.previous_rejects+1
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
                self.end_event.set()
        await self.say('OK. Please listen carefully. It is important that you do this math task entirely in your head and by yourself - do not write anything down or use any devices like a calculator as that would invalidate the results.')
        await self.say('I will ask you to count as accurately as you are able by subtracting a number from another starting number, and then keep subtracting the number from your answer until you reach zero or I tell you to stop.')
        await self.say('For example, if I asked you to subtract the number five starting from fifty you would first enter fourty five  then fourty then thirty five and so on.')
        await self.ask_permission('Are you ready?', end_label='math_intro')
        number_choice = random.sample(nineties, k=len(nineties))
        self.call_log.miscellaneous['countdown'] =  await self.ask(
            f'Okay. Subtract number three starting from number {number_choice}. Enter your answers on the keypad and press the star key after each answer. Say continue when you are done. Begin after the beep.',
            wait_time=30,
            stopword_list=['continue'],
            file=os.path.join(sounds_directory,"beep.wav"),
            dtmf_wait_character='*0*',
        )
        self.call_log.rejected = 'completed'
        await self.ask('Thank you. This concludes our session. Thank you for your patience in completing this longer baseline call. Future calls after today will be shorter with fewer and shorter explanations. Please remember that you can call Anna to take this assessment at any time, if you miss a scheduled call or want to report any symptoms. Until next time. Good-bye. ', wait_time=0.2)
        self.end_event.set()

    async def participant_initiated_script(self):
        self.call_log.miscellaneous['direction'] = 'inbound'
        if self.previous_calls > 9999:
            await self.say("Hello! My name is Anna. I'm an automated nursing assistant that is part of a research study. This call will be recorded for research use.", start_label="short_intro")
            await self.say("If this is an emergency, please hang up and call nine one one. If you need non-emergency medical attention, please call your provider's nursing line.")
            await self.short_script()
            return
        else:
            await self.say("Hello! My name is Anna. I'm an automated nursing assistant that is part of a research study. This call is not monitored in real time. ", start_label="long_intro")
            await self.say("If this is an emergency, please hang up and call nine one one. If you need non-emergency medical attention, please call your provider's nursing line.")
            await self.long_script()
            return

    async def short_script(self):
        print('began short script')
        #this parameter is set by default until the participant gets through the script
        self.call_log.rejected = "ignored"

        print('finished intro')
        #this prepares the llm for the conversation
        await self.ask_permission("Are you ready to talk?", end_label="short_intro")
        self.call_log.rejected = "accepted"
        #free conversation
        init_question = ("Okay. First, tell me about any new headaches, fuzzy thinking, hallucinations, or difficulty with finding words. If no changes to report, describe any other new symptoms or just talk about how you have been doing in general in the past few hours. Please begin after the beep.")
        await self.ask(init_question, 
                                start_label="how_are_you_doing",
                                end_label="how_are_you_doing",
                                forward_to_llm=True,
                                minimum_turn_time=10,
                                file=os.path.join(sounds_directory,"beep.wav"),
                                await_silence=True,
                                silence_window=2, stopword_list=['continue'])
        await self.memory_test_loop('short')
        await self.say('Thank you. Now, the math task. Please switch to speakerphone and bring up the keypad. ', start_label='math_intro')
        first_pound = await self.ask("Press the pound key on your phone's keypad when you are ready. ", wait_time=10, dtmf_wait_character='#')
        #this will happen if no digits were pressed
        if first_pound == True:
           second_pound = await self.ask("Please press the pound key on your phone's keypad when you are ready. ", wait_time=10, dtmf_wait_character='#')
           if second_pound == True:
                current_hour = datetime.now().hour
                rejects = self.call_log.previous_rejects+1
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
                await self.end_event.wait()
        number_choice = random.sample(nineties, k=len(nineties))
        self.call_log.miscellaneous['countdown'] =  await self.ask(
            f'OK. Subtract number three starting from {number_choice}. Remember to press the star key after each response. Begin subtracting after the beep.',
            wait_time=30,
            dtmf_wait_character='*0*',
            stopword_list=['continue'],
            file=os.path.join(sounds_directory,"beep.wav")
        )
        self.call_log.rejected = 'completed'
        await self.ask("Thank you. This concludes our session. Please remember that you can also call Anna to take this assessment at any time. Until next time. Goodbye.", wait_time=0.1)
        self.end_event.set()


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
    
    async def memory_test_loop(self, version):
        tasks = []
        if version == 'long':
            await self.long_memory_test()
        elif version == 'short':
            await self.short_memory_test()
        if 'repeat'in self.call_log.memory_exercise_reply.lower():
            await self.repeat_memory_test()

    

    async def long_memory_test(self):
        #word list recall task
        #word list recall task
        await self.say("First, let’s test your memory. I will give you eight words. Please concentrate and repeat each word you hear aloud. Later, I will ask you to recall all eight words.  ",
                       end_label="memory_permission")
        await self.say('If you get distracted or cannot hear the words clearly, just say repeat and I will give you a different set of words to remember.')
        await self.ask_permission("Are you ready?", 
                        end_label="memory_description")
        await self.say("Here is the list.", start_label="memory_begin", end_label="memory_begin")
        word_files = random.sample(MEMORY_WORDS, k=8)
        self.call_log.memory_exercise_words = [w.split('.')[0] for w in word_files]
        for w in word_files:
            await self.say(file=os.path.join(word_recordings_directory,w), final_pause=1.0, initial_pause=1.0, start_label="memory_word", end_label="memory_word")
        self.call_log.memory_exercise_reply = await self.ask("Now repeat as many of these words as you remember and say continue when you are done. If you got distracted, or interrupted, or couldn't hear the words clearly, say the word, repeat. Please begin after the beep.",
                            file=os.path.join(sounds_directory,"beep.wav"), wait_time=30,  stopword_list=['continue', 'repeat'],
                            start_label="memory_response", end_label="memory_response")        


    async def short_memory_test(self):
        #word list recall task
        await self.say("Thank you. Now, on to the memory task. I will give you eight words to remember. Please concentrate and say aloud each word after you hear it. ",
                       end_label="memory_permission")
        await self.ask_permission("Are you ready?", 
                        end_label="memory_description")
        await self.say("Here is the list.", start_label="memory_begin", end_label="memory_begin")
        word_files = random.sample(MEMORY_WORDS, k=8)
        self.call_log.memory_exercise_words = [w.split('.')[0] for w in word_files]
        for w in word_files:

            await self.say(file=os.path.join(word_recordings_directory,w), final_pause=1.0, initial_pause=1.0, start_label="memory_word", end_label="memory_word")
        self.call_log.memory_exercise_reply = await self.ask("Now repeat as many of these words as you remember and say continue when you are done, or repeat if you got distracted, or interrupted, or couldn't hear the words clearly. Please begin after the beep.",
                            file=os.path.join(sounds_directory,"beep.wav"), wait_time=30,  stopword_list=['continue', 'repeat'],
                            start_label="memory_response", end_label="memory_response")        


    async def repeat_memory_test(self):
        await self.say("Here is another list.")
        word_files_2 = random.sample(MEMORY_WORDS, k=8)
        self.call_log.miscellaneous['memory_exercise_words_2'] = [w.split('.')[0] for w in word_files_2]
        for w in word_files_2:
            await self.say(file=os.path.join(word_recordings_directory,w), final_pause=1.0, initial_pause=1.0, start_label="memory_word_2", end_label="memory_word_2")
        self.call_log.memory_exercise_reply_2 = await self.ask("Now repeat as many of these words as you remember and say continue when you are done. Please begin after the beep.",
                            file=os.path.join(sounds_directory,"beep.wav"), wait_time=30,  stopword_list=['continue'],
                            start_label="memory_response_2", end_label="memory_response_2")  



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
