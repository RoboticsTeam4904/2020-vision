use rmp_serde;
use serde::Serialize;
use std::{
    io::Result,
    net::{SocketAddr, UdpSocket},
};

pub struct UdpSender {
    socket: UdpSocket,
}

impl UdpSender {
    pub fn new(src_port: u16, dst_address: String) -> UdpSender {
        let socket = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], src_port))).unwrap();
        socket.connect(dst_address).unwrap();
        socket.set_nonblocking(true).unwrap();
        UdpSender { socket }
    }
    pub fn send(&self, msg: impl Serialize) -> Result<usize> {
        let buf = rmp_serde::to_vec(&msg).unwrap();
        self.socket.send(&buf)
    }
}
