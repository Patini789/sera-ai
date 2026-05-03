const { Client, GatewayIntentBits } = require('discord.js');
const { joinVoiceChannel, createAudioPlayer, createAudioResource, EndBehaviorType } = require('@discordjs/voice');
const prism = require('prism-media');
const dgram = require('dgram');
const express = require('express');

const PYTHON_UDP_PORT = 5005; 
const udpClient = dgram.createSocket('udp4');
const app = express();
const activeStreams = new Set();
app.use(express.json());

const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] });
let currentConnection = null;
let audioPlayer = createAudioPlayer();

client.on('ready', () => console.log(`Node connected to Discord as: ${client.user.tag}`));

app.post('/join', (req, res) => {
    const { channelId, guildId } = req.body;
    const guild = client.guilds.cache.get(guildId);
    
    currentConnection = joinVoiceChannel({
        channelId: channelId,
        guildId: guildId,
        adapterCreator: guild.voiceAdapterCreator,
        selfDeaf: false,
    });

    currentConnection.subscribe(audioPlayer);

    // speaking
    currentConnection.receiver.speaking.on('start', (userId) => {
        const user = client.users.cache.get(userId);
        if (user?.bot) return; // Ignorar a otros bots
        if (user?.bot || activeStreams.has(userId)) return;
        activeStreams.add(userId);


        const audioStream = currentConnection.receiver.subscribe(userId, {
            end: { behavior: EndBehaviorType.AfterSilence, duration: 200 },
        });

        const decoder = new prism.opus.Decoder({ rate: 16000, channels: 1, frameSize: 960 });
        
        audioStream.pipe(decoder).on('data', (chunk) => {
            const nameBuf = Buffer.from(user.username);
            const header = Buffer.alloc(1);
            header.writeUInt8(nameBuf.length);
            
            const finalPacket = Buffer.concat([header, nameBuf, chunk]);
            udpClient.send(finalPacket, PYTHON_UDP_PORT, '127.0.0.1');
        });
        audioStream.on('end', () => {
        activeStreams.delete(userId);
    });
    });

    res.send({ status: "ok" });
});

app.post('/play', (req, res) => {
    const { filepath } = req.body;
    if (currentConnection) {
        const resource = createAudioResource(filepath);
        audioPlayer.play(resource);
    }
    res.send({ status: "playing" });
});

app.post('/leave', (req, res) => {
    if (currentConnection) currentConnection.destroy();
    res.send({ status: "left" });
});

// Discord Token
client.login(''); 
app.listen(3000, () => console.log('🚀 Microservicio de Voz corriendo en puerto local 3000'));