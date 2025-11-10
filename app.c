volatile unsigned int *const UART_DR = (unsigned int *)0x101f1000; // the address to match the 'versatilepb' board's UART0
void print_str(const char *s){
    while(*s != '\0'){
	    *UART_DR = *s;
	    s++;
    }
}

int main(void){
    print_str("This is the test code, print string via UART");
	while (1){};
    return 0;
}

void UART0_Handler(void){
    print_str("--- UART Interrupted Occured! --- \n");
}
