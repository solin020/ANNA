from starlette.responses import Response
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.background import BackgroundTask
from starlette.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Connect
import os, logging, sys
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from .call_state import CallState, wss_url
from .database import engine, CallLog, ScheduledCall, Participant, Job
import secrets
from datetime import datetime, timedelta, timezone
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session, select
import asyncio
from dataclasses import dataclass
from sqlmodel import SQLModel
import base64
from typing import Union, Annotated
from ..config import frontend_directory, call_recordings_directory, gateway_username, gateway_password, anna_segmenter_directory, anna_segmenter_assets
from starlette.responses import FileResponse 
import phonenumbers
import subprocess
import re
from .parse_atq import parse_atq
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
import json


dir_path = os.path.dirname(os.path.realpath(__file__))

#region authentication

app = FastAPI()

class NoCacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

app.add_middleware(NoCacheMiddleware)

origins = ['*',]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

async def get_current_username(request: Request) -> HTTPBasicCredentials:

    credentials = await security(request)

    correct_username = secrets.compare_digest(getattr(credentials, "username", "na"), gateway_username)
    correct_password = secrets.compare_digest(getattr(credentials, "password", "na"), gateway_password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


logger = logging.getLogger("gateway")

#endregion




#region Twilio Logic
call_dict:dict[str,CallState] = {}

@app.route('/make-call', methods=['POST'])
async def make_call(request):
    phone_number, previous_rejects = await request.json()
    task = BackgroundTask(begin_stream_conversation, phone_number, int(previous_rejects))
    return Response('OK', media_type='text/plain', background=task)

async def begin_stream_conversation(number, previous_rejects):
    call =  await CallState.outbound_call(number, previous_rejects)
    call_dict[call.call_sid] = call




@app.route('/stream-conversation-receive', methods=['POST',])
async def stream_conversation_receive(request):
    form = await request.form()
    vr = VoiceResponse()
    call_sid = form['CallSid']
    call_dict[call_sid] = await CallState.inbound_call(form)
    connect = Connect()
    connect.stream(url=f'{wss_url}/stream-conversation-socket')
    vr.append(connect)
    return Response(vr.__str__(), media_type='application/xml')

@app.websocket_route('/stream-conversation-socket')
async def stream_conversation_socket(websocket):
    from .call_state import EndCall
    print('websocket recieved')
    await websocket.accept()
    call_sid = ""
    stream_sid = ""
    try:
        while True:
            frame = await websocket.receive_json()
            if frame['event'] == 'connected':
                print('connection accepted by stream_conversation', flush=True)
            elif frame['event'] == 'start':
                stream_sid = frame['start']['streamSid']
                call_sid = frame['start']['callSid']
            elif frame['event'] == 'dtmf':
                print('got dtmf', frame['dtmf']['digit'])
                if call_sid in call_dict:
                    call_state = call_dict[call_sid]
                    await call_state.controller.dtmf_queue.put(frame['dtmf']['digit'])
            elif frame['event'] == 'media':
                if call_sid in call_dict:
                    call_state = call_dict[call_sid]
                    await call_state.handle_streams(frame, websocket, stream_sid)
                #This handles the case where the CallState object for this call hasn't been initialized yet
                else:
                    print('not recieved', call_sid)
                    recieved_bytes = base64.b64decode(frame['media']['payload'])
                    inbytes = b'\x7f' * len(recieved_bytes)
                    if inbytes:
                        media_data = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                            "payload": base64.b64encode(inbytes).decode('utf-8')
                            }
                        }
                        await websocket.send_json(media_data) 
            elif frame['event'] == 'stop':
                print('starting close')
                await websocket.close()
                print('connection closed gracefully', flush=True)
                break
    except (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK):
        print('connection closed disgracefully', flush=True)
    except EndCall:
        print('call ended')

@app.post('/call-status')
async def call_status(r:Request):
    form = await r.form()
    call_sid = form['CallSid'] #type:ignore
    status = form['CallStatus']
    print('status', status)
    if status not in ('queued', 'in-progress', 'completed'):
        print('abnormal status', status)
        call_state = call_dict.pop(call_sid)#type:ignore
        call_state.call_log.rejected=status
        call_state.after_call()#type:ignore
    else:
        call_dict[call_sid].answered.set()#type:ignore

#endregion

#region Call scheduling Logic

async def scheduled_call_task(start_time:datetime, phone_number:str, rejects:int):
    st = start_time.astimezone(datetime.now(timezone.utc).astimezone().tzinfo)
    command = f"""echo '{sys.executable} {dir_path}/../make_call.py {phone_number} {rejects}' |at {st.hour:0>2}:{st.minute:0>2} {st.month:0>2}/{st.day:0>2}/{st.year}"""
    print('at', command)
    proc = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    print('at', stdout, stderr)
    m = re.search(r'job \d+', stderr.decode('utf8'))
    jobid = m.group(0).split(' ')[-1]
    with Session(engine) as s:
        t = ScheduledCall(id=jobid,time=st, phone_number=phone_number, rejects=rejects)
        s.add(t)
        s.commit()





@app.post('/schedule-call', )
async def schedule_call(t:ScheduledCall):
    await scheduled_call_task(t.time, t.phone_number, 0)
    return Response('OK', media_type='text/plain')



async def schedule_calls(p:Participant):
    for day in range((p.end_date-p.start_date).days+1):
        for hour, minute in zip((p.morning_time, p.afternoon_time, p.evening_time),
                (p.morning_minute, p.afternoon_minute, p.evening_minute)):
            print(day,hour,minute)
            time = p.start_date + timedelta(days=day, hours=hour, minutes=minute)
            time = time.astimezone(timezone.utc)
            await scheduled_call_task(time, p.phone_number, 0)



@app.post('/schedule_calls',)
async def post_schedule_calls(phone_number:str, start_date:datetime, end_date:datetime, 
        morning_time:int, afternoon_time:int,evening_time:int,
        morning_minute:int, afternoon_minute:int,evening_minute:int):
    p = Participant(start_date=start_date, end_date=end_date, phone_number=phone_number, 
                   morning_time=morning_time, afternoon_time=afternoon_time, evening_time=evening_time,
                   morning_minute=morning_minute, afternoon_minute=afternoon_minute, evening_minute=evening_minute)
    await schedule_calls(p)
    return Response('OK', media_type='text/plain')
    

@app.post('/cancel-call',)
async def cancel_call(t:ScheduledCall):
    jobid = t.id
    try:
        with Session(engine) as s:
            statement = select(ScheduledCall).where(
                ScheduledCall.id == jobid
            ).where(ScheduledCall.phone_number == t.phone_number)
            results = s.exec(statement)
            r = results.one()
            if r.time>datetime.now():
                try:
                    os.system(f'at -r {jobid}')
                except Exception as e:
                    print(e)
            s.delete(r)
            s.commit()
    except NoResultFound:
        pass

@app.post('/cancel-calls',)
async def cancel_schedule_calls(phone_number: str):
    with Session(engine) as s:
        statement = select(ScheduledCall).where(ScheduledCall.phone_number == phone_number)
        results = s.exec(statement).all()
        for call in results:
            if call.time > datetime.now():
                try:
                    jobid = call.id
                    os.system(f'at -r {jobid}')
                except Exception as e:
                    print(e)
            s.delete(call)
        s.commit()


@app.get('/list-scheduled-calls',)
async def list_schedule_calls(phone_number: str) -> list[ScheduledCall]:
    with Session(engine) as s:
        statement = select(ScheduledCall).where(ScheduledCall.phone_number == phone_number).where(ScheduledCall.time > datetime.now()).order_by(ScheduledCall.time)
        results = s.exec(statement)
        return results.all()

@app.get('/entire-schedule',)
async def entire_schedule() -> list[Job]:
    return await parse_atq()


@app.post('/add-participant',)
async def add_participant(p: Participant):
    pn = phonenumbers.parse(p.phone_number, "US")
    #this accounts for variability in phone number formatting
    adjusted_pn = f'+{pn.country_code}{pn.national_number}'
    p.phone_number = adjusted_pn
    await schedule_calls(p)
    with Session(engine) as s:
        s.add(p)
        s.commit()
    return Response('OK', media_type='text/plain')

@app.post('/delete-participant',)
async def delete_participant(participant_study_id: str, phone_number:str):
    with Session(engine) as s:
        p_statement = select(Participant).where(Participant.phone_number==phone_number).where(Participant.participant_study_id==participant_study_id)
        p = s.exec(p_statement).one()
        sc_statements = select(ScheduledCall).where(ScheduledCall.phone_number == phone_number)
        cl_statements = select(CallLog).where(CallLog.phone_number == phone_number)
        sc_results = s.exec(sc_statements).all()
        for call in sc_results:
            if call.time > datetime.now():
                try:
                    jobid = call.id
                    os.system(f'at -r {jobid}')
                except Exception as e:
                    print(e)
            s.delete(call)
        cl_results = s.exec(cl_statements)
        for cl in cl_results:
            s.delete(cl)
        s.delete(p)
        s.commit()
    return Response('OK', media_type='text/plain')






#endregion

#region Web Portal

@app.get('/api/call-log',)
async def api_call_log(call_sid:str):
    with Session(engine) as s:
        statement = select(CallLog).filter(CallLog.call_sid == call_sid)
        result = s.exec(statement).one()
    return result

@dataclass
class CallLogHeader(SQLModel):
    call_sid:str
    participant_study_id:str
    timestamp: datetime

@app.get('/api/call-list',)
async def api_call_list(participant_study_id:str) -> list[CallLogHeader]:
    with Session(engine) as s:
        print(f'{participant_study_id=}')
        participant = s.exec(select(Participant).where(Participant.participant_study_id == participant_study_id)).one()
        statement = select(CallLog.call_sid, CallLog.participant_study_id, CallLog.timestamp).where(CallLog.phone_number == participant.phone_number).order_by(CallLog.timestamp)
        call_logs = s.exec(statement).all()
        return [CallLogHeader(
                call_sid=call_sid, participant_study_id=participant_study_id, timestamp=timestamp
            ) 
            for call_sid, participant_study_id, timestamp in call_logs
        ]

@app.get('/api/participant-list',)
async def api_participant_list() -> list[Participant]:
    with Session(engine) as s:
        statement = select(Participant)
        return s.exec(statement).all()







@app.get('/anna-segmenter/list-mp3s')
async def list_wavs():
    with Session(engine) as s:
        call_logs = [cl for cl in s.exec(
                select(CallLog.call_sid, CallLog.participant_study_id, CallLog.timestamp, CallLog.rejected).order_by(CallLog.participant_study_id, CallLog.timestamp)
            ).all()
            if (cl.participant_study_id != 'JacobSolinsky') and ('test' not in cl.participant_study_id ) and (cl.rejected == 'completed')
        ] 
    retdict = {f'bot_{cl.call_sid}':
        {'participant_study_id':cl.participant_study_id, 'timestamp':str(cl.timestamp), 'audio':False, 'autoSegmentation':False,'manualSegmentation': False} 
        for cl in call_logs
    }
    for f in os.listdir(call_recordings_directory):
        prefix = f.split('.')[0]
        if prefix not in retdict:
            continue
        if f.endswith('.mp3') or f.endswith('.wav'):
            retdict[prefix]['audio'] = f
        elif f.endswith('.auto.json'):
            retdict[prefix]['autoSegmentation'] = f
        elif f.endswith('.manual.json'):
            retdict[prefix]['manualSegmentation'] = f
    return retdict


@app.route('/anna-segmenter/save-manual-segmentation', methods=['POST'])
async def save_manual_segmentation(r:Request):
    form = await r.form()
    filenamePrefix = form['filenamePrefix']
    manualSegmentationJson = form['manualSegmentationJson']
    with open(f'{call_recordings_directory}/{filenamePrefix}.manual.json', 'w+') as f:
        f.write(manualSegmentationJson)
    with open(f'{call_recordings_directory}/{filenamePrefix}.manual.seg', 'w+') as f:
        for line in json.loads(manualSegmentationJson):
            label = line['label']
            start = line['start']
            end = line['end']
            f.write(f'{label}\t{start}\t{end}\n')
    return Response('OK')
    

@app.route('/anna-segmenter/index.html', methods=['GET'])
async def read_anna_segmenter(r:Request):
    return FileResponse(os.path.join(anna_segmenter_directory, 'index.html'))

@app.route('/anna-segmenter/vite.svg', methods=['GET'])
async def read_anna_svg(r:Request):
    return FileResponse(os.path.join(anna_segmenter_directory, 'vite.svg'))

@app.get("/", )
async def read_index1():
    return FileResponse(os.path.join(frontend_directory, 'dist/index.html'))

@app.get("/favicon.ico")
async def read_index2():
    return FileResponse(os.path.join(frontend_directory, 'dist/favicon.ico'))

app.mount("/assets", StaticFiles(directory=os.path.join(frontend_directory, 'dist/assets')), name="dist")
app.mount("/src", StaticFiles(directory=os.path.join(frontend_directory, 'src')), name="src")
app.mount("/recordings", StaticFiles(directory=call_recordings_directory), name="recordings")
app.mount("/anna-segmenter/recordings", StaticFiles(directory=call_recordings_directory), name="recordings")
app.mount("/anna-segmenter/assets", StaticFiles(directory=anna_segmenter_assets), name="recordings")
app.mount("/", StaticFiles(directory=os.path.join(frontend_directory, 'public')), name="public") 
#endregion

