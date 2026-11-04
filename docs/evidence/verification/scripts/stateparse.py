"""Parse PyBoy v7 / v30 DMG state files into named fields, using the gates in pyboy/core/*.py."""
import struct
def u64(b): return struct.unpack('<Q',b)[0]
def parse(raw, wram=None):
    d={}; ver=raw[0]; d['version']=ver
    if ver==7:
        o=1; d['bootrom']=raw[o];o+=1
    elif ver==30:
        d['bootrom']=raw[1]; d['key0'],d['key1'],d['double_speed'],d['cgb']=raw[2:6];o=6
    else: raise ValueError(ver)
    for k in 'A F B C D E'.split(): d[k]=raw[o];o+=1
    for k in 'HL SP PC'.split(): d[k]=raw[o]|raw[o+1]<<8;o+=2
    if ver==7:
        d['IME']=raw[o];d['halted']=raw[o+1];d['stopped']=raw[o+2];d['IE']=raw[o+3];o+=4
    else:
        d['IME']=raw[o];d['ie_delay']=raw[o+1];d['halted']=raw[o+2];d['stopped']=raw[o+3];d['IE']=raw[o+4];d['int_queued']=raw[o+5];d['IF']=raw[o+6];o+=7
        d['cpu_cycles']=u64(raw[o:o+8]);o+=8;d['speed_switch_halt']=u64(raw[o:o+8]);o+=8
        d['oam_dma']=raw[o:o+4].hex();o+=4;d['oam_dma_start']=u64(raw[o:o+8]);o+=8
    d['VRAM']=raw[o:o+0x2000];o+=0x2000;d['OAM']=raw[o:o+0xA0];o+=0xA0
    for k in 'LCDC BGP OBP0 OBP1 STAT LY LYC SCY SCX WY WX'.split(): d[k]=raw[o];o+=1
    if ver==30:
        d['obj_prio']=raw[o];o+=1
        d['scanline']=raw[o:o+720];o+=720
        for k in 'lcd_cgb downgraded speed_shift frame_done first_frame lcd_reset'.split(): d[k]=raw[o];o+=1
        for k in 'lcd_last_cycles lcd_clock lcd_clock_target'.split(): d[k]=u64(raw[o:o+8]);o+=8
        d['next_stat_mode']=raw[o];d['stat_transition']=raw[o+1];o+=2
        sound_start=o
        # locate WRAM: renderer is 144*160*5+1 before ram; use known tail length instead
        # tail = ram 8491 + wram_select 1 + timer 25 + cart(4+32768) + interaction 2 + serial 36 + sgb ?
        i=raw.find(wram[:4096],o) if wram else -1
        assert i>0,'wram not located'
        d['sound']=raw[o:i-115201]; d['sound_offset']=o
        d['screenbuffer']=raw[i-115201:i-1]; d['wy_activated_frame']=raw[i-1]; o=i
    else:
        o+=720
        d['screenbuffer']=raw[o:o+92160];o+=92160
    d['WRAM']=raw[o:o+8192];o+=8192
    d['NONIO0']=raw[o:o+0x60];o+=0x60;d['IOPORTS']=raw[o:o+0x4C];o+=0x4C;d['HRAM']=raw[o:o+0x7F];o+=0x7F
    if ver==7:
        o+=0x20;d['obj_prio']=raw[o];o+=4;d['wram_select']=raw[o];o+=0x10
        d['DIV']=raw[o];d['TIMA']=raw[o+1];d['DIV_counter']=raw[o+2]|raw[o+3]<<8;d['TIMA_counter']=raw[o+4]|raw[o+5]<<8;d['TMA']=raw[o+6];d['TAC']=raw[o+7];o+=8
    else:
        d['wram_select']=raw[o];o+=1
        d['DIV']=raw[o];d['TIMA']=raw[o+1];d['DIV_counter']=raw[o+2]|raw[o+3]<<8;d['TIMA_counter']=raw[o+4]|raw[o+5]<<8;d['TMA']=raw[o+6];d['TAC']=raw[o+7];d['tima_reload']=raw[o+8];o+=9
        d['timer_last_cycles']=u64(raw[o:o+8]);o+=8;d['timer_legacy']=u64(raw[o:o+8]);o+=8
    d['rombank'],d['rambank'],d['ram_enabled'],d['memmodel']=raw[o:o+4];o+=4
    d['SRAM']=raw[o:o+32768];o+=32768
    d['joy_dir'],d['joy_std']=raw[o:o+2];o+=2
    if ver==30:
        d['SB'],d['SC'],d['ser_en'],d['ser_int']=raw[o:o+4];o+=4
        d['serial_rest']=raw[o:o+32];o+=32
        d['sgb']=raw[o:];o=len(raw)
    assert o==len(raw),(o,len(raw))
    return d
